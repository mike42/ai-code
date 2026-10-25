package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"ai-code/internal/tool"
)

// ProtocolVersion is the newest revision this client speaks; it accepts a
// server that answers with any of supportedVersions.
const ProtocolVersion = "2025-06-18"

var supportedVersions = []string{"2025-06-18", "2025-03-26", "2024-11-05"}

// Tool is one tool as the server describes it. InputSchema is kept exactly
// as received, so the request carrying it stays byte-identical.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
	Annotations struct {
		ReadOnlyHint bool `json:"readOnlyHint"`
	} `json:"annotations"`
}

// Client is a connected, initialised MCP server.
type Client struct {
	conn       *Conn
	ServerName string
	Version    string
}

// Connect runs binary with args, never through a shell, and initialises it.
func Connect(ctx context.Context, binary string, args, env []string, notify func(method string)) (*Client, error) {
	c := newConn()
	setNotify(c, notify)
	if _, err := startProcess(c, binary, args, env); err != nil {
		return nil, err
	}
	return initialise(ctx, c, nil)
}

// Dial connects to a Streamable HTTP server and initialises it.
func Dial(ctx context.Context, url string, headers map[string]string, notify func(method string)) (*Client, error) {
	c := newConn()
	setNotify(c, notify)
	t := newHTTPTransport(c, url, headers)
	return initialise(ctx, c, t.setVersion)
}

func setNotify(c *Conn, notify func(string)) {
	if notify != nil {
		c.notify = func(method string, _ json.RawMessage) { notify(method) }
	}
}

func initialise(ctx context.Context, c *Conn, agreed func(string)) (*Client, error) {
	var res struct {
		ProtocolVersion string `json:"protocolVersion"`
		ServerInfo      struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"serverInfo"`
	}
	err := c.Call(ctx, "initialize", map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "ai-code", "version": "1"},
	}, &res)
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("handshake: %w", err)
	}
	if !slices.Contains(supportedVersions, res.ProtocolVersion) {
		c.Close()
		return nil, fmt.Errorf("the server speaks MCP %q; ai-code speaks %s",
			res.ProtocolVersion, strings.Join(supportedVersions, ", "))
	}
	if agreed != nil {
		agreed(res.ProtocolVersion)
	}
	if err := c.Notify(ctx, "notifications/initialized", nil); err != nil {
		c.Close()
		return nil, err
	}
	return &Client{conn: c, ServerName: res.ServerInfo.Name, Version: res.ProtocolVersion}, nil
}

// Tools lists every tool, following pagination.
func (cl *Client) Tools(ctx context.Context) ([]Tool, error) {
	var all []Tool
	cursor := ""
	for {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		var page struct {
			Tools      []Tool `json:"tools"`
			NextCursor string `json:"nextCursor"`
		}
		if err := cl.conn.Call(ctx, "tools/list", params, &page); err != nil {
			return nil, fmt.Errorf("listing tools: %w", err)
		}
		all = append(all, page.Tools...)
		if page.NextCursor == "" {
			return all, nil
		}
		cursor = page.NextCursor
	}
}

type content struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
	URI      string `json:"uri"`
	Name     string `json:"name"`
	Resource *struct {
		URI  string `json:"uri"`
		Text string `json:"text"`
	} `json:"resource"`
}

// Call runs one tool. A failure of the tool, or of the server, comes back as
// an error result the model can read; only cancellation returns ctx's error.
func (cl *Client) Call(ctx context.Context, name string, args json.RawMessage, progress func(string)) tool.Result {
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	var res struct {
		Content           []content       `json:"content"`
		StructuredContent json.RawMessage `json:"structuredContent"`
		IsError           bool            `json:"isError"`
	}
	err := cl.conn.call(ctx, "tools/call", map[string]any{"name": name, "arguments": args}, &res, progress)
	if err != nil {
		if ctx.Err() != nil {
			return tool.Result{Content: "Cancelled.", IsError: true, Interrupted: true}
		}
		return tool.Errorf("%v", err)
	}
	out := tool.Result{IsError: res.IsError}
	var text []string
	for _, c := range res.Content {
		switch c.Type {
		case "text":
			text = append(text, c.Text)
		case "image":
			data, err := base64.StdEncoding.DecodeString(c.Data)
			if err != nil {
				text = append(text, "[an image that could not be decoded]")
				continue
			}
			out.Images = append(out.Images, tool.Image{MIME: c.MimeType, Data: data})
		case "resource_link":
			text = append(text, strings.TrimSpace("[resource] "+c.Name+" "+c.URI))
		case "resource":
			if c.Resource != nil && c.Resource.Text != "" {
				text = append(text, c.Resource.Text)
			} else if c.Resource != nil {
				text = append(text, "[resource] "+c.Resource.URI)
			}
		default:
			text = append(text, "["+c.Type+" content, which ai-code does not show]")
		}
	}
	if len(text) == 0 && len(res.StructuredContent) > 0 {
		text = append(text, string(res.StructuredContent))
	}
	out.Content = strings.Join(text, "\n")
	return out
}

// Err is why the connection ended, or nil while it is up.
func (cl *Client) Err() error { return cl.conn.Err() }

// Close stops the server, or ends the HTTP session.
func (cl *Client) Close() error { return cl.conn.Close() }
