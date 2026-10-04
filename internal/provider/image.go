package provider

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"slices"
)

// Image is a picture shown to the model alongside a message's text.
type Image struct {
	MIME   string `json:"mime"`
	Data   []byte `json:"data"`
	Width  int    `json:"width,omitempty"`
	Height int    `json:"height,omitempty"`
}

// NewImage reads the type and size from the data itself.
func NewImage(data []byte) Image {
	img := Image{MIME: http.DetectContentType(data), Data: data}
	if cfg, _, err := image.DecodeConfig(bytes.NewReader(data)); err == nil {
		img.Width, img.Height = cfg.Width, cfg.Height
	}
	return img
}

// Tokens is a starting estimate of what the image costs in context: one token
// per 28×28 pixels, as Qwen's vision encoders spend. The server's reported
// usage replaces it after the first request that carries the image.
func (i Image) Tokens() int {
	w, h := i.Width, i.Height
	if w <= 0 || h <= 0 {
		w, h = 1024, 1024
	}
	return max(1, w*h/(28*28))
}

func (i Image) dataURL() string {
	return "data:" + i.MIME + ";base64," + base64.StdEncoding.EncodeToString(i.Data)
}

type wirePart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *wireImageURL `json:"image_url,omitempty"`
}

type wireImageURL struct {
	URL string `json:"url"`
}

func contentParts(text string, images []Image) []wirePart {
	parts := []wirePart{{Type: "text", Text: text}}
	for _, img := range images {
		parts = append(parts, wirePart{Type: "image_url", ImageURL: &wireImageURL{URL: img.dataURL()}})
	}
	return parts
}

// ImageRequest asks an image model for one picture. An empty Size leaves the
// dimensions to the server.
type ImageRequest struct {
	Model  string
	Prompt string
	Size   string
}

// ImageGenerator is implemented by backends that serve image models.
type ImageGenerator interface {
	GenerateImage(ctx context.Context, req ImageRequest) (Image, error)
}

// IsImageModel reports whether a catalogue entry says it makes images. A
// backend that labels nothing is taken at its word that the model exists.
func IsImageModel(m ModelInfo) bool {
	return len(m.Labels) == 0 || slices.Contains(m.Labels, "image")
}

func (c *OpenAI) GenerateImage(ctx context.Context, req ImageRequest) (Image, error) {
	fields := map[string]any{"model": req.Model, "prompt": req.Prompt, "n": 1, "response_format": "b64_json"}
	if req.Size != "" {
		fields["size"] = req.Size
	}
	body, err := json.Marshal(fields)
	if err != nil {
		return Image{}, err
	}
	resp, err := c.doWithRetry(ctx, "POST", "/images/generations", body, withKey)
	if err != nil {
		return Image{}, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return Image{}, err
	}
	var out struct {
		Data []struct {
			B64 string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return Image{}, fmt.Errorf("parsing the image response: %w", err)
	}
	if len(out.Data) == 0 || out.Data[0].B64 == "" {
		return Image{}, fmt.Errorf("the server answered without an image")
	}
	data, err := base64.StdEncoding.DecodeString(out.Data[0].B64)
	if err != nil {
		return Image{}, fmt.Errorf("decoding the image: %w", err)
	}
	return NewImage(data), nil
}
