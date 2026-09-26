package tool

import (
	"bufio"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// ignoreRule is one line from a .gitignore file.
type ignoreRule struct {
	pattern  string
	negate   bool
	dirOnly  bool
	anchored bool
	// base is the directory the rule was declared in, relative to the walk root.
	base string
}

// ignoreSet implements enough of gitignore to keep a code search out of
// node_modules, build output and vendored trees.
//
// It is a subset, not a reimplementation of git's matcher: anchoring, directory
// suffixes, negation, basename patterns and `**` are handled; the rarer corners
// are not. Shelling out to `git check-ignore` would be exact but costs a
// process per path, and reimplementing the whole thing is a project of its own.
// Being slightly over-inclusive in a search is a much cheaper error than being
// slow.
type ignoreSet struct {
	rules []ignoreRule
}

func (s *ignoreSet) addFile(gitignorePath, base string) {
	f, err := os.Open(gitignorePath)
	if err != nil {
		return
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		r := ignoreRule{base: base}
		if strings.HasPrefix(line, "!") {
			r.negate = true
			line = line[1:]
		}
		if strings.HasSuffix(line, "/") {
			r.dirOnly = true
			line = strings.TrimSuffix(line, "/")
		}
		if strings.HasPrefix(line, "/") {
			r.anchored = true
			line = strings.TrimPrefix(line, "/")
		} else if strings.Contains(line, "/") {
			// A pattern with an interior slash is anchored to its own directory.
			r.anchored = true
		}
		if line == "" {
			continue
		}
		r.pattern = line
		s.rules = append(s.rules, r)
	}
}

// match reports whether relPath (slash-separated, relative to the walk root) is
// ignored. Later rules win, which is how negation works.
func (s *ignoreSet) match(relPath string, isDir bool) bool {
	ignored := false
	for _, r := range s.rules {
		if r.dirOnly && !isDir {
			continue
		}
		target := relPath
		if r.base != "" {
			if !strings.HasPrefix(relPath, r.base+"/") {
				continue
			}
			target = strings.TrimPrefix(relPath, r.base+"/")
		}
		if r.matches(target) {
			ignored = !r.negate
		}
	}
	return ignored
}

func (r ignoreRule) matches(relPath string) bool {
	pattern := r.pattern

	if strings.HasPrefix(pattern, "**/") {
		pattern = strings.TrimPrefix(pattern, "**/")
		return matchAnyDepth(pattern, relPath)
	}
	if !r.anchored {
		return matchAnyDepth(pattern, relPath)
	}

	if ok, _ := path.Match(pattern, relPath); ok {
		return true
	}
	// An anchored directory pattern also ignores everything beneath it.
	if strings.HasPrefix(relPath, pattern+"/") {
		return true
	}
	// Handle a `**` in the middle by matching each side.
	if before, after, found := strings.Cut(pattern, "/**/"); found {
		if strings.HasPrefix(relPath, before+"/") {
			return matchAnyDepth(after, strings.TrimPrefix(relPath, before+"/"))
		}
	}
	return false
}

// matchAnyDepth matches a basename-style pattern against any path segment, and
// against any directory prefix so that ignoring `build` also ignores
// `build/x/y.go`.
func matchAnyDepth(pattern, relPath string) bool {
	segments := strings.Split(relPath, "/")
	for i, seg := range segments {
		if ok, _ := path.Match(pattern, seg); ok {
			// A match on a non-final segment means an ancestor directory is
			// ignored, which ignores everything under it.
			_ = i
			return true
		}
	}
	return false
}

// walkOptions controls the shared directory walker used by grep and glob.
type walkOptions struct {
	root       string
	respectGit bool
	includeAll bool // include dotfiles
	maxFiles   int
	// namedHidden are dot-prefixed names the caller asked for by name, such
	// as the ".devcontainer" in a `**/.devcontainer/**` glob. Hidden files
	// are skipped by default because nobody means them by `*.go`, but a
	// pattern that spells one out has said what it means, and answering "no
	// matches" to it is a lie about the filesystem.
	namedHidden map[string]bool
}

// hiddenSkipped reports whether a dot-prefixed name should be passed over.
func (o walkOptions) hiddenSkipped(name string) bool {
	return !o.includeAll && strings.HasPrefix(name, ".") && !o.namedHidden[name]
}

// alwaysSkip are directories no code search should ever descend into, whether
// or not a .gitignore mentions them. .git in particular contains packed objects
// that will happily match any regex.
var alwaysSkip = map[string]bool{
	".git": true, ".hg": true, ".svn": true, ".jj": true,
}

// walkFiles visits every file under root that survives the ignore rules,
// calling fn for each. It loads .gitignore files as it descends, so a nested
// ignore file applies to its own subtree.
func walkFiles(opts walkOptions, fn func(absPath, relPath string) error) error {
	ig := &ignoreSet{}
	if opts.respectGit {
		ig.addFile(filepath.Join(opts.root, ".gitignore"), "")
		// A global excludes file is common; honour it when present.
		if home, err := os.UserHomeDir(); err == nil {
			ig.addFile(filepath.Join(home, ".config", "git", "ignore"), "")
		}
	}

	count := 0
	return filepath.WalkDir(opts.root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			// An unreadable directory is not a reason to abandon the search.
			if d != nil && d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if p == opts.root {
			return nil
		}

		name := d.Name()
		relPath, relErr := filepath.Rel(opts.root, p)
		if relErr != nil {
			return nil
		}
		relPath = filepath.ToSlash(relPath)

		if d.IsDir() {
			if alwaysSkip[name] {
				return filepath.SkipDir
			}
			if opts.hiddenSkipped(name) {
				return filepath.SkipDir
			}
			if opts.respectGit {
				ig.addFile(filepath.Join(p, ".gitignore"), relPath)
				if ig.match(relPath, true) {
					return filepath.SkipDir
				}
			}
			return nil
		}

		if opts.hiddenSkipped(name) {
			return nil
		}
		if opts.respectGit && ig.match(relPath, false) {
			return nil
		}

		count++
		if opts.maxFiles > 0 && count > opts.maxFiles {
			return filepath.SkipAll
		}
		return fn(p, relPath)
	})
}

// DirIgnorer returns a test for whether one entry of dir is ignored, for
// callers that read a single directory instead of walking the tree.
//
// Tab completion is that caller: walking a large repository on a keystroke is
// not an option, but offering a path that grep and glob will never look at is
// worse than offering nothing. The ignore files of root and of every directory
// between root and dir are loaded, which is the subset of walkFiles' behaviour
// that a single directory can observe.
func DirIgnorer(root, dir string) func(name string, isDir bool) bool {
	never := func(string, bool) bool { return false }

	rel, err := filepath.Rel(root, dir)
	if err != nil {
		return never
	}
	rel = filepath.ToSlash(rel)
	if rel == ".." || strings.HasPrefix(rel, "../") {
		return never
	}
	if rel == "." {
		rel = ""
	}

	ig := &ignoreSet{}
	ig.addFile(filepath.Join(root, ".gitignore"), "")
	if home, err := os.UserHomeDir(); err == nil {
		ig.addFile(filepath.Join(home, ".config", "git", "ignore"), "")
	}
	base := ""
	for _, seg := range strings.Split(rel, "/") {
		if seg == "" {
			continue
		}
		base = path.Join(base, seg)
		ig.addFile(filepath.Join(root, filepath.FromSlash(base), ".gitignore"), base)
	}

	return func(name string, isDir bool) bool {
		if alwaysSkip[name] {
			return true
		}
		return ig.match(path.Join(rel, name), isDir)
	}
}
