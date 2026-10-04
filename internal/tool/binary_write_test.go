package tool

import (
	"bytes"
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var pictureBytes = []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe")

func writeBinary(t *testing.T, st *State, path string, data []byte) Result {
	t.Helper()
	return (&WriteTool{}).Run(context.Background(), st, mustJSON(t, map[string]string{
		"path": path, "content_base64": base64.StdEncoding.EncodeToString(data)}))
}

func TestBinaryContentsAreWrittenExactly(t *testing.T) {
	dir := t.TempDir()
	res := writeBinary(t, NewState(dir), "img/logo.png", pictureBytes)
	if res.IsError {
		t.Fatal(res.Content)
	}
	got, err := os.ReadFile(filepath.Join(dir, "img", "logo.png"))
	if err != nil || !bytes.Equal(got, pictureBytes) {
		t.Errorf("the file holds %q, want the bytes sent", got)
	}
	if !strings.Contains(res.Content, "Created img/logo.png") {
		t.Errorf("result: %s", res.Content)
	}
}

func TestOnePictureCanReplaceAnotherWithoutARead(t *testing.T) {
	st, path := fixture(t, "logo.png", string(pictureBytes))
	newer := append([]byte{}, pictureBytes...)
	newer[len(newer)-1] = 0x01
	if res := writeBinary(t, st, path, newer); res.IsError {
		t.Fatalf("a picture cannot be read, so demanding a read first blocks it forever: %s", res.Content)
	}
}

func TestBinaryContentsDoNotOverwriteAnUnreadTextFile(t *testing.T) {
	st, path := fixture(t, "README.md", "# Project\n\nNotes the model never saw.\n")
	res := writeBinary(t, st, path, pictureBytes)
	if !res.IsError {
		t.Fatal("a text file was replaced by a picture without being read first")
	}
	if got, _ := os.ReadFile(path); !strings.Contains(string(got), "never saw") {
		t.Error("the text file was changed")
	}
}

func TestBadBase64IsRefused(t *testing.T) {
	res := (&WriteTool{}).Run(context.Background(), NewState(t.TempDir()), mustJSON(t, map[string]string{
		"path": "x.png", "content_base64": "not base64!"}))
	if !res.IsError {
		t.Error("invalid base64 was written")
	}
}
