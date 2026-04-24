package api

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

const requestIDHeader = "X-Request-Id"

var defaultAllowedOrigins = []string{
	"http://localhost:*",
	"http://127.0.0.1:*",
	"app://ai-engine",
}

func (s *Server) requestIDMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := strings.TrimSpace(c.GetHeader(requestIDHeader))
		if requestID == "" {
			requestID = fmt.Sprintf("%d", time.Now().UnixNano())
		}
		c.Set("request_id", requestID)
		c.Header(requestIDHeader, requestID)
		c.Next()
	}
}

func (s *Server) corsMiddleware() gin.HandlerFunc {
	cfg := s.config.Server.CORS
	allowedHeaders := strings.Join(cfg.AllowedHeaders, ", ")
	if allowedHeaders == "" {
		allowedHeaders = "Content-Type, Authorization"
	}

	return func(c *gin.Context) {
		if !cfg.Enabled {
			c.Next()
			return
		}

		origin := c.GetHeader("Origin")
		if origin == "" {
			c.Next()
			return
		}

		if !originAllowed(origin, cfg.AllowedOrigins) {
			c.AbortWithStatus(http.StatusForbidden)
			return
		}

		c.Header("Access-Control-Allow-Origin", origin)
		c.Header("Access-Control-Allow-Headers", allowedHeaders)
		c.Header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Vary", "Origin")

		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

func originAllowed(origin string, allowed []string) bool {
	// When allowed is nil (field omitted), fall back to defaults.
	// When allowed is a non-nil empty slice (explicitly configured as []),
	// treat it as "no origins allowed".
	if allowed == nil {
		allowed = defaultAllowedOrigins
	}
	for _, pattern := range allowed {
		if originMatchesPattern(origin, pattern) {
			return true
		}
	}
	return false
}

func originMatchesPattern(origin, pattern string) bool {
	origin = strings.TrimSpace(origin)
	pattern = strings.TrimSpace(pattern)
	if origin == "" || pattern == "" {
		return false
	}
	if origin == pattern {
		return true
	}
	if !strings.HasSuffix(pattern, ":*") {
		return false
	}

	trimmed := strings.TrimSuffix(pattern, ":*")
	allowedURL, err := url.Parse(trimmed)
	if err != nil || allowedURL.Scheme == "" || allowedURL.Host == "" {
		return false
	}
	originURL, err := url.Parse(origin)
	if err != nil || originURL.Scheme == "" || originURL.Host == "" {
		return false
	}
	return originURL.Scheme == allowedURL.Scheme && originURL.Hostname() == allowedURL.Hostname()
}
