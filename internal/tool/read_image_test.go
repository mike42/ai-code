package tool

import (
	"bytes"
	"context"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadingAnImageAttachesIt(t *testing.T) {
	dir := t.TempDir()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 40, 30))); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "capture.png"), buf.Bytes(), 0o644)

	res := (&ReadTool{MaxBytes: 10000, MaxLines: 100}).Run(context.Background(), NewState(dir),
		[]byte(`{"path":"capture.png"}`))
	if res.IsError {
		t.Fatalf("read refused the image: %s", res.Content)
	}
	if len(res.Images) != 1 {
		t.Fatalf("got %d images, want the file attached", len(res.Images))
	}
	img := res.Images[0]
	if img.MIME != "image/png" || img.Width != 40 || img.Height != 30 || !bytes.Equal(img.Data, buf.Bytes()) {
		t.Errorf("image = %s %dx%d (%d bytes), want the PNG as read", img.MIME, img.Width, img.Height, len(img.Data))
	}
	if !strings.Contains(res.Content, "40x30 image/png") {
		t.Errorf("content = %q, want it described", res.Content)
	}
}

func TestOtherBinaryFilesAreStillRefused(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "blob"), []byte{0, 1, 2, 3, 0xff}, 0o644)
	res := (&ReadTool{MaxBytes: 10000, MaxLines: 100}).Run(context.Background(), NewState(dir),
		[]byte(`{"path":"blob"}`))
	if !res.IsError || len(res.Images) != 0 {
		t.Errorf("binary non-image: %+v", res)
	}
}
