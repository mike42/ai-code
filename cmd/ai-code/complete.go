package main

import (
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"ai-code/internal/provider"
	"ai-code/internal/tool"
)

// complete supplies Tab candidates for the line up to the cursor: slash
// command names, model ids after /model, and otherwise a path.
//
// The offset it returns is where the candidates replace the line, so a path
// can be completed in the middle of a sentence. That is the whole reason file
// completion exists here: there is no @file sigil and nothing is attached by
// typing a path, but a path spelled the way the tools expect saves the model
// the turns it would otherwise spend hunting for the file with ls and grep.
func (a *App) complete(line string) (int, []string) {
	name, argAt, arg, ok := commandContext(line)
	if !ok {
		return completePath(a.cwd, a.projectRoot, line)
	}
	if argAt < 0 {
		// Every prefix match, not matchCommands': "/mode" dispatches to /mode
		// but must still offer /model, which is the one being typed.
		var out []string
		for _, c := range commands {
			if strings.HasPrefix(c.name, name) {
				out = append(out, "/"+c.name+" ")
			}
		}
		return 0, out
	}
	if m := matchCommands(name); len(m) == 1 && m[0].name == "model" {
		return argAt, a.modelCandidates(arg)
	}
	return 0, nil
}

// commandContext splits a slash-command line into the command name and its
// argument, with argAt -1 while the name itself is still being typed.
//
// A word with a further slash in it is a path someone began the line with, not
// a command: "/usr/local/..." must still complete as a file.
func commandContext(line string) (name string, argAt int, arg string, ok bool) {
	if !strings.HasPrefix(line, "/") {
		return "", 0, "", false
	}
	rest := line[1:]
	i := strings.IndexByte(rest, ' ')
	if i < 0 {
		if strings.Contains(rest, "/") {
			return "", 0, "", false
		}
		return strings.ToLower(rest), -1, "", true
	}
	name = rest[:i]
	if strings.Contains(name, "/") {
		return "", 0, "", false
	}
	trimmed := strings.TrimLeft(line[1+i:], " ")
	return strings.ToLower(name), len(line) - len(trimmed), trimmed, true
}

// modelCandidates offers the model ids of the current provider.
func (a *App) modelCandidates(prefix string) []string {
	if a.client == nil {
		return nil
	}
	// Under .nocloud a cloud model is not merely refused when selected: its
	// name is never shown, so completion cannot become the route by which a
	// tree that forbids cloud ends up on one.
	if a.noCloud && a.client.Class() == provider.ClassCloud {
		return nil
	}
	// Tab never reaches the network. A keystroke that occasionally blocks on a
	// provider round trip is worse than one that occasionally completes
	// nothing, so only the on-disk catalogue is read -- expired included,
	// since the names in it are still the best guess available and /model
	// checks the chosen one against the provider anyway.
	cached, _ := loadModelCache(a.client.Name())
	if cached == nil {
		return nil
	}
	var out []string
	for _, m := range cached.Models {
		if strings.HasPrefix(m.ID, prefix) {
			out = append(out, m.ID)
		}
	}
	sort.Strings(out)
	return out
}

// completePath completes the last whitespace-delimited word as a path,
// relative to the working directory, one segment at a time.
//
// One directory read per Tab, never a tree walk: in a repository of any size
// the walk is the difference between a completion and a pause. A name
// containing a space completes only as far as its first space, which is
// visibly nothing rather than quietly the wrong path.
func completePath(cwd, root, line string) (int, []string) {
	start := strings.LastIndexAny(line, " \t") + 1
	word := line[start:]
	// An empty word lists the working directory, as a shell does, whether the
	// line is bare or the cursor sits after a trailing space. Tab means nothing
	// else in this editor, so either press is someone reaching for a filename;
	// answering one and ignoring the other would be a rule to remember rather
	// than a saved keystroke. The listing costs a screen at worst -- it is
	// capped, and it prints above the prompt without touching what is typed.
	dir, base := path.Split(word)
	readDir := filepath.FromSlash(dir)
	if !path.IsAbs(word) {
		readDir = filepath.Join(cwd, readDir)
	}
	entries, err := os.ReadDir(readDir)
	if err != nil {
		return 0, nil
	}
	ignored := tool.DirIgnorer(root, readDir)

	var out []string
	for _, ent := range entries {
		name := ent.Name()
		if !strings.HasPrefix(name, base) {
			continue
		}
		// Dotfiles appear only once asked for by name, as in a shell.
		if strings.HasPrefix(name, ".") && !strings.HasPrefix(base, ".") {
			continue
		}
		isDir := ent.IsDir()
		if !isDir && ent.Type()&os.ModeSymlink != 0 {
			if st, err := os.Stat(filepath.Join(readDir, name)); err == nil {
				isDir = st.IsDir()
			}
		}
		if ignored(name, isDir) {
			continue
		}
		if isDir {
			name += "/"
		}
		out = append(out, dir+name)
	}
	sort.Strings(out)
	return start, out
}
