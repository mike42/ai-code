package main

import (
	"fmt"
	"sort"
	"strings"
)

type toolChoice struct {
	known      []string
	offDefault map[string]bool
	enabled    map[string]bool
}

func splitToolList(v string) []string {
	return strings.FieldsFunc(v, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' })
}

func resolveTools(known []string, offDefault map[string]bool, enable, disable []string) (*toolChoice, error) {
	c := &toolChoice{known: append([]string(nil), known...), offDefault: offDefault, enabled: map[string]bool{}}
	sort.Strings(c.known)
	isKnown := map[string]bool{}
	for _, n := range c.known {
		isKnown[n] = true
		c.enabled[n] = !offDefault[n]
	}
	on := map[string]bool{}
	for _, n := range enable {
		if !isKnown[n] {
			return nil, fmt.Errorf("--enable-tools: there is no tool named %q. %s", n, c.describe())
		}
		on[n] = true
		c.enabled[n] = true
	}
	for _, n := range disable {
		if !isKnown[n] {
			return nil, fmt.Errorf("--disable-tools: there is no tool named %q. %s", n, c.describe())
		}
		if on[n] {
			return nil, fmt.Errorf("%s is both enabled and disabled; name it in one of --enable-tools and --disable-tools", n)
		}
		c.enabled[n] = false
	}
	return c, nil
}

func (c *toolChoice) describe() string {
	var on, off []string
	for _, n := range c.known {
		if c.offDefault[n] {
			off = append(off, n)
		} else {
			on = append(on, n)
		}
	}
	return fmt.Sprintf("Tools: %s; off unless enabled: %s.", strings.Join(on, ", "), strings.Join(off, ", "))
}

func (c *toolChoice) allow(name string) bool { return c.enabled[name] }

func (c *toolChoice) among(names []string) []string {
	var out []string
	for _, n := range names {
		if c.enabled[n] {
			out = append(out, n)
		}
	}
	return out
}
