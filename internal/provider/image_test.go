package provider

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func tinyPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewRGBA(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestPicturesAreSentAsPartsOfTheMessage(t *testing.T) {
	pic := NewImage(tinyPNG(t, 3, 2))
	body := encodeBody(t, Options{}, Request{Model: "m", Messages: []Message{
		{Role: RoleTool, ToolCallID: "c1", Name: "generate_image", Content: "Saved logo.png.", Images: []Image{pic}},
		{Role: RoleUser, Content: "plain"},
	}})
	msgs := body["messages"].([]any)
	parts, ok := msgs[0].(map[string]any)["content"].([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("a message with a picture should be text then the picture, got %#v", msgs[0])
	}
	if parts[0].(map[string]any)["text"] != "Saved logo.png." {
		t.Errorf("the text part is %#v", parts[0])
	}
	url := parts[1].(map[string]any)["image_url"].(map[string]any)["url"].(string)
	if want := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pic.Data); url != want {
		t.Errorf("the picture is sent as %.60s..., want a PNG data URL", url)
	}
	if msgs[1].(map[string]any)["content"] != "plain" {
		t.Errorf("a message without pictures should stay a plain string, got %#v", msgs[1])
	}
}

func TestNewImageReadsTheTypeAndSizeFromTheData(t *testing.T) {
	img := NewImage(tinyPNG(t, 56, 28))
	if img.MIME != "image/png" || img.Width != 56 || img.Height != 28 {
		t.Errorf("got %s %dx%d, want image/png 56x28", img.MIME, img.Width, img.Height)
	}
	if img.Tokens() != 2 {
		t.Errorf("a 56x28 picture is estimated at %d tokens, want 2", img.Tokens())
	}
}

func TestGenerateImageAsksForOnePictureAndDecodesIt(t *testing.T) {
	pic := tinyPNG(t, 4, 4)
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" || r.Method != "POST" {
			http.Error(w, "wrong endpoint "+r.Method+" "+r.URL.Path, 404)
			return
		}
		json.NewDecoder(r.Body).Decode(&got)
		json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{
			"b64_json": base64.StdEncoding.EncodeToString(pic)}}})
	}))
	defer srv.Close()
	c, err := NewOpenAI(Options{Name: "t", BaseURL: srv.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}

	img, err := c.GenerateImage(context.Background(), ImageRequest{Model: "flux", Prompt: "a fox"})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(img.Data, pic) || img.Width != 4 {
		t.Errorf("the picture came back as %d bytes, %dx%d", len(img.Data), img.Width, img.Height)
	}
	if got["model"] != "flux" || got["prompt"] != "a fox" || got["n"] != float64(1) || got["response_format"] != "b64_json" {
		t.Errorf("the request was %v", got)
	}
	if _, ok := got["size"]; ok {
		t.Error("with no size asked for, the server's own size should be left alone")
	}

	if _, err := c.GenerateImage(context.Background(), ImageRequest{Model: "flux", Prompt: "a fox", Size: "512x512"}); err != nil {
		t.Fatal(err)
	}
	if got["size"] != "512x512" {
		t.Errorf("size was sent as %v", got["size"])
	}
}

func TestGenerateImageSaysWhenNoPictureCameBack(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"data":[]}`))
	}))
	defer srv.Close()
	c, _ := NewOpenAI(Options{Name: "t", BaseURL: srv.URL})
	_, err := c.GenerateImage(context.Background(), ImageRequest{Model: "flux", Prompt: "x"})
	if err == nil || !strings.Contains(err.Error(), "without an image") {
		t.Errorf("got %v", err)
	}
}
