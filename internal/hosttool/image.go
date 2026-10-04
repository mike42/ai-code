package hosttool

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"path"
	"regexp"
	"strings"
	"time"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

const ImageName = "generate_image"

// ImageSource is the session's current provider, as an image generator. It
// is asked on every call, so a switch of provider moves image generation too.
type ImageSource func(ctx context.Context) (provider.ImageGenerator, error)

// Images adds generate_image to inner. The picture is made from this machine;
// the file is saved wherever inner's write tool saves, so a container or
// remote machine gets the file and never talks to the image model.
type Images struct {
	inner  tool.Executor
	model  string
	source ImageSource
}

func WithImages(inner tool.Executor, model string, source ImageSource) *Images {
	return &Images{inner: inner, model: model, source: source}
}

const imageSchema = `{
  "type": "object",
  "properties": {
    "prompt": {"type": "string", "description": "What to draw: subject, style, composition, colours and any text in the picture. The image model sees only this, not the conversation."},
    "path": {"type": "string", "description": "Where to save the picture, absolute or relative to the working directory, ending in .png."},
    "size": {"type": "string", "description": "WIDTHxHEIGHT in pixels, such as 1024x1024. Leave it out for the model's usual size."}
  },
  "required": ["prompt", "path"]
}`

var sizePattern = regexp.MustCompile(`^[1-9][0-9]{1,4}x[1-9][0-9]{1,4}$`)

func (e *Images) description() string {
	return "Draws a picture with the image model " + e.model + " and saves it as a file. " +
		"You are shown the picture, so look at it and generate again with a better prompt " +
		"if it is not what was wanted. Saving to an existing path replaces that picture."
}

func (e *Images) Definitions() []tool.Definition {
	return append(e.inner.Definitions(), tool.Definition{
		Name: ImageName, Description: e.description(), Schema: json.RawMessage(imageSchema),
	})
}

func (e *Images) Execute(ctx context.Context, req tool.Request) (tool.Result, error) {
	if req.Name != ImageName {
		return e.inner.Execute(ctx, req)
	}
	start := time.Now()
	res := e.generate(ctx, req)
	res.Duration = time.Since(start).Round(time.Millisecond).String()
	return res, nil
}

type imageArgs struct {
	Prompt string `json:"prompt"`
	Path   string `json:"path"`
	Size   string `json:"size"`
}

func (e *Images) generate(ctx context.Context, req tool.Request) tool.Result {
	var a imageArgs
	if r := decode(req.Args, &a); r != nil {
		return *r
	}
	a.Size = strings.ToLower(strings.TrimSpace(a.Size))
	switch {
	case strings.TrimSpace(a.Prompt) == "":
		return tool.Errorf("prompt is empty; describe the picture to draw.")
	case strings.TrimSpace(a.Path) == "":
		return tool.Errorf("path is empty; say where to save the picture, such as images/logo.png.")
	case a.Size != "" && !sizePattern.MatchString(a.Size):
		return tool.Errorf("size %q is not WIDTHxHEIGHT, such as 1024x1024; leave it out for the model's usual size.", a.Size)
	}

	gen, err := e.source(ctx)
	if err != nil {
		return tool.Errorf("no picture was made: %v", err)
	}
	img, err := gen.GenerateImage(ctx, provider.ImageRequest{Model: e.model, Prompt: a.Prompt, Size: a.Size})
	if err != nil {
		if ctx.Err() != nil {
			return tool.Result{IsError: true, Interrupted: true, Content: "Interrupted before the picture was finished; nothing was saved."}
		}
		return tool.Errorf("%s could not make the picture: %v", e.model, err)
	}

	args, _ := json.Marshal(map[string]string{"path": a.Path, "content_base64": base64.StdEncoding.EncodeToString(img.Data)})
	saved, err := e.inner.Execute(ctx, tool.Request{CallID: req.CallID, Name: "write", Args: args})
	if err != nil {
		return tool.Errorf("the picture was made but could not be saved: %v", err)
	}
	if saved.IsError {
		saved.Content = "The picture was made but not saved: " + saved.Content
		return saved
	}

	what := fmt.Sprintf("%dx%d %s", img.Width, img.Height, img.MIME)
	note := ""
	if want := extensionFor(img.MIME); want != "" && !strings.EqualFold(path.Ext(a.Path), want) {
		note = fmt.Sprintf(" The picture is %s data, so a name ending in %s would describe it better.", img.MIME, want)
	}
	return tool.Result{
		Content: fmt.Sprintf("%s It is %s, drawn by %s, and is shown below.%s", saved.Content, what, e.model, note),
		Display: strings.TrimSuffix(saved.Display, ")") + fmt.Sprintf(", %s, %s)", what, e.model),
		Images:  []tool.Image{{MIME: img.MIME, Data: img.Data, Width: img.Width, Height: img.Height}},
		Files:   saved.Files,
		Cwd:     saved.Cwd,
	}
}

func extensionFor(mime string) string {
	switch mime {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	case "image/gif":
		return ".gif"
	}
	return ""
}

func (e *Images) Unwrapped() tool.Executor {
	if u, ok := e.inner.(interface{ Unwrapped() tool.Executor }); ok {
		return u.Unwrapped()
	}
	return e.inner
}

func (e *Images) Fork() (tool.Executor, error) { return e.inner.Fork() }

func (e *Images) IsReadOnly(name string) bool {
	if name == ImageName {
		return false
	}
	if ro, ok := e.inner.(interface{ IsReadOnly(string) bool }); ok {
		return ro.IsReadOnly(name)
	}
	return false
}

func (e *Images) SetProgress(p tool.Progress) {
	if pr, ok := e.inner.(tool.ProgressReporter); ok {
		pr.SetProgress(p)
	}
}

func (e *Images) Close() error {
	if c, ok := e.inner.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}
