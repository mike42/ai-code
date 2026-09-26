// Package audit holds tests that check properties of the whole program rather
// than of any one package.
package audit

import (
	"os/exec"
	"strings"
	"testing"
)

// ai-code makes no network connection except to the provider it is configured
// with. That is a promise in the README, and a promise nobody checks is a
// promise that quietly stops being true. These tests are the check.

func TestNoAnalyticsDependencies(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "./...").Output()
	if err != nil {
		t.Skipf("go list unavailable: %v", err)
	}

	// Substrings that appear in the import paths of the usual telemetry and
	// analytics SDKs.
	banned := []string{
		"sentry", "datadog", "segment", "amplitude", "mixpanel", "posthog",
		"bugsnag", "rollbar", "newrelic", "honeycomb", "opentelemetry",
		"go.opencensus.io", "analytics", "telemetry", "statsd", "prometheus",
	}
	for _, line := range strings.Split(string(out), "\n") {
		pkg := strings.ToLower(strings.TrimSpace(line))
		if pkg == "" {
			continue
		}
		for _, b := range banned {
			if strings.Contains(pkg, b) {
				t.Errorf("dependency %q looks like telemetry; ai-code does not phone home", line)
			}
		}
	}
}

// TestOnlyTheProviderPackageSpeaksHTTP keeps outbound network access in one
// place. If some other package acquires an HTTP client, that is where an
// unnoticed phone-home would live, and this test is how it gets noticed.
func TestOnlyTheProviderPackageSpeaksHTTP(t *testing.T) {
	out, err := exec.Command("go", "list", "-f",
		`{{.ImportPath}} {{join .Imports " "}}`, "./...").Output()
	if err != nil {
		t.Skipf("go list unavailable: %v", err)
	}

	allowed := map[string]bool{
		"ai-code/internal/provider": true, // the only package that may reach the network
		"ai-code/internal/audit":    true, // this test package
	}
	networkPkgs := []string{"net/http", "net/url", "net"}

	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		pkg := fields[0]
		if allowed[pkg] || strings.HasSuffix(pkg, "_test") {
			continue
		}
		for _, imp := range fields[1:] {
			for _, n := range networkPkgs {
				if imp == n {
					t.Errorf("%s imports %s; outbound network access belongs in "+
						"internal/provider so there is exactly one place to audit it", pkg, imp)
				}
			}
		}
	}
}

// TestNoBackgroundUpdateCheck guards against the other common phone-home: a
// version check on startup. It is also a startup-latency problem.
func TestNoBackgroundUpdateCheck(t *testing.T) {
	// The audit package is excluded because these very patterns live in it.
	out, err := exec.Command("grep", "-rn", "-i",
		"-e", "update.check", "-e", "checkForUpdate", "-e", "latest_version",
		"--include=*.go", "--exclude-dir=audit", "../..").Output()
	if err != nil {
		return // grep exits non-zero when nothing matches, which is the pass.
	}
	if s := strings.TrimSpace(string(out)); s != "" {
		t.Errorf("found what looks like an update check:\n%s", s)
	}
}
