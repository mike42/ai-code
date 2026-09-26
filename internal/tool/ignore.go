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
// node_modules and vendored trees. A subset, erring toward over-inclusion.
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

// matchAnyDepth matches a basename-style pattern against any path segment; a
// match on any ancestor ignores everything beneath it.
func matchAnyDepth(pattern, relPath string) bool {
	segments := strings.Split(relPath, "/")
	for i, seg := range segments {
		if ok, _ := path.Match(pattern, seg); ok {
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
	// namedHidden are dot-prefixed names the caller asked for by name; a
	// pattern that spells one out has said what it means.
	namedHidden map[string]bool
}

// hiddenSkipped reports whether a dot-prefixed name should be passed over.
func (o walkOptions) hiddenSkipped(name string) bool {
	return !o.includeAll && strings.HasPrefix(name, ".") && !o.namedHidden[name]
}

// alwaysSkip are directories no code search should descend into, gitignored or
// not. .git holds packed objects that match almost any regex.
var alwaysSkip = map[string]bool{
	".git": true, ".hg": true, ".svn": true, ".jj": true,
}

// walkFiles calls fn for every file under root that survives the ignore rules,
// loading nested .gitignore files as it descends.
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
// callers that read a single directory instead of walking the tree. It loads
// the ignore files from root down to dir.
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
