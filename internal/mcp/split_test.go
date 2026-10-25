package mcp

import (
	"reflect"
	"strings"
	"testing"
)

func TestSplit(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{`/opt/mcp/hello`, []string{"/opt/mcp/hello"}},
		{`  a   b	c  `, []string{"a", "b", "c"}},
		{`/opt/mcp/hello --greeting "good morning" --root=/my\ dir`,
			[]string{"/opt/mcp/hello", "--greeting", "good morning", "--root=/my dir"}},
		{`'it''s'`, []string{"its"}},
		{`'a "b" \c'`, []string{`a "b" \c`}},
		{`"say \"hi\" \\ \n"`, []string{`say "hi" \ \n`}},
		{`--root="/my dir"`, []string{"--root=/my dir"}},
		{`a'b'"c"d`, []string{"abcd"}},
		{`x '' ""`, []string{"x", "", ""}},
		{`C:\tools\hello.exe --dir C:\work`, []string{`C:\tools\hello.exe`, "--dir", `C:\work`}},
		{`'\\host\share'`, []string{`\\host\share`}},
		{`\'quoted\'`, []string{"'quoted'"}},
		{`a\\b`, []string{`a\b`}},
		{`$HOME *.go ~ ; | & < > ` + "`x`" + ` #c`,
			[]string{"$HOME", "*.go", "~", ";", "|", "&", "<", ">", "`x`", "#c"}},
		{``, nil},
		{`   `, nil},
		{`trailing\`, []string{`trailing\`}},
		{`héllo "wörld"`, []string{"héllo", "wörld"}},
	}
	for _, c := range cases {
		got, err := Split(c.in)
		if err != nil {
			t.Errorf("Split(%q): %v", c.in, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("Split(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestSplitUnterminated(t *testing.T) {
	cases := map[string]string{
		`hello "good morning`: `unterminated " at column 7`,
		`a 'b`:                `unterminated ' at column 3`,
		`"a\"`:                `unterminated " at column 1`,
	}
	for in, want := range cases {
		_, err := Split(in)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("Split(%q) error = %v, want %q", in, err, want)
		}
	}
}
