package hosttool

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

type fakeImages struct {
	asked []provider.ImageRequest
	data  []byte
	err   error
}

func (f *fakeImages) GenerateImage(_ context.Context, req provider.ImageRequest) (provider.Image, error) {
	f.asked = append(f.asked, req)
	if f.err != nil {
		return provider.Image{}, f.err
	}
	return provider.NewImage(f.data), nil
}

func pngOf(t *testing.T, w, h int) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewRGBA(image.Rect(0, 0, w, h))); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func imageExec(t *testing.T, gen *fakeImages) (*Images, string) {
	t.Helper()
	dir := t.TempDir()
	inner := tool.NewLocalExecutor(tool.NewState(dir), &tool.WriteTool{}, &tool.LsTool{})
	return WithImages(inner, "Flux-Test", func(context.Context) (provider.ImageGenerator, error) { return gen, nil }), dir
}

func runImage(t *testing.T, e *Images, args string) tool.Result {
	t.Helper()
	res, err := e.Execute(context.Background(), tool.Request{CallID: "c1", Name: ImageName, Args: json.RawMessage(args)})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func TestThePictureIsSavedWhereTheToolsRunAndShownToTheModel(t *testing.T) {
	gen := &fakeImages{data: pngOf(t, 8, 6)}
	e, dir := imageExec(t, gen)
	if !hasDef(e, ImageName) || !hasDef(e, "write") {
		t.Fatalf("definitions: %+v", e.Definitions())
	}

	res := runImage(t, e, `{"prompt":"a heron","path":"art/heron.png","size":"512x384"}`)
	if res.IsError {
		t.Fatal(res.Content)
	}
	saved, err := os.ReadFile(filepath.Join(dir, "art", "heron.png"))
	if err != nil || !bytes.Equal(saved, gen.data) {
		t.Fatalf("the workspace holds %d bytes (%v), want the picture's %d", len(saved), err, len(gen.data))
	}
	if len(gen.asked) != 1 || gen.asked[0] != (provider.ImageRequest{Model: "Flux-Test", Prompt: "a heron", Size: "512x384"}) {
		t.Errorf("the image model was asked %+v", gen.asked)
	}
	if len(res.Images) != 1 || !bytes.Equal(res.Images[0].Data, gen.data) || res.Images[0].Width != 8 {
		t.Errorf("the model is not shown the picture: %+v", res.Images)
	}
	if !strings.Contains(res.Content, "8x6 image/png") || !strings.Contains(res.Display, "art/heron.png") {
		t.Errorf("content %q, display %q", res.Content, res.Display)
	}
}

func TestBadCallsAreRefusedBeforeTheImageModelIsAsked(t *testing.T) {
	gen := &fakeImages{data: pngOf(t, 2, 2)}
	e, _ := imageExec(t, gen)
	for _, args := range []string{
		`{"prompt":"","path":"a.png"}`,
		`{"prompt":"a fox","path":" "}`,
		`{"prompt":"a fox","path":"a.png","size":"big"}`,
		`{"prompt":"a fox","path":"a.png","size":"1024*1024"}`,
		`{"prompt":`,
	} {
		if res := runImage(t, e, args); !res.IsError {
			t.Errorf("%s was accepted", args)
		}
	}
	if len(gen.asked) != 0 {
		t.Errorf("the image model was asked %d times for bad calls", len(gen.asked))
	}
}

func TestAPictureThatCannotBeSavedSaysSo(t *testing.T) {
	gen := &fakeImages{data: pngOf(t, 2, 2)}
	e, dir := imageExec(t, gen)
	os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("unread text\n"), 0o644)
	res := runImage(t, e, `{"prompt":"a fox","path":"notes.txt"}`)
	if !res.IsError || !strings.Contains(res.Content, "made but not saved") || len(res.Images) != 0 {
		t.Errorf("got %+v", res)
	}
}

func TestAnImageModelErrorReachesTheModel(t *testing.T) {
	gen := &fakeImages{err: errors.New("model not found")}
	e, _ := imageExec(t, gen)
	res := runImage(t, e, `{"prompt":"a fox","path":"fox.png"}`)
	if !res.IsError || !strings.Contains(res.Content, "Flux-Test could not make the picture: model not found") {
		t.Errorf("got %q", res.Content)
	}
}

func TestAWrongExtensionIsPointedOut(t *testing.T) {
	e, _ := imageExec(t, &fakeImages{data: pngOf(t, 2, 2)})
	res := runImage(t, e, `{"prompt":"a fox","path":"fox.jpg"}`)
	if res.IsError || !strings.Contains(res.Content, "ending in .png") {
		t.Errorf("got %q", res.Content)
	}
}

func TestOtherToolsPassThrough(t *testing.T) {
	e, _ := imageExec(t, &fakeImages{})
	res, err := e.Execute(context.Background(), tool.Request{Name: "ls", Args: json.RawMessage(`{}`)})
	if err != nil || res.IsError {
		t.Errorf("ls: %v %+v", err, res)
	}
}
