package api

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func getBuiltinMCPTools() []string {
	return append([]string(nil), "mcp.describe_connection", "mcp.echo")
}

func (s *Server) handleCapabilities(c *gin.Context) {
	health := s.supervisor.Health()
	contextReady := false
	if contextHealth, ok := health["context"].(map[string]interface{}); ok {
		if ready, ok := contextHealth["ready"].(bool); ok {
			contextReady = ready
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"execution_mode": s.supervisor.ExecutionMode(),
		"services": gin.H{
			"runtime": gin.H{
				"ready":     serviceReady(health),
				"streaming": "sse",
			},
			"rag": gin.H{
				"ready": serviceReady(health) && contextReady,
			},
			"context": gin.H{
				"ready": contextReady,
			},
			"training": gin.H{
				"staged":      true,
				"active_runs": nestedInt(health, "training", "active_runs"),
			},
			"mcp": gin.H{
				"staged":        true,
				"connections":   nestedInt(health, "mcp", "connections"),
				"builtin_tools": getBuiltinMCPTools(),
			},
		},
	})
}

func (s *Server) handleTrainingStatus(c *gin.Context) {
	health := s.supervisor.Health()
	c.JSON(http.StatusOK, gin.H{
		"staged":      true,
		"active_runs": nestedInt(health, "training", "active_runs"),
		"message":     "training controls are exposed as staged status in the frontend-ready MVP",
	})
}

func (s *Server) handleMCPStatus(c *gin.Context) {
	health := s.supervisor.Health()
	c.JSON(http.StatusOK, gin.H{
		"staged":        true,
		"connections":   nestedInt(health, "mcp", "connections"),
		"builtin_tools": getBuiltinMCPTools(),
		"message":       "MCP transports are staged; only built-in plumbing tools are executable",
	})
}

func serviceReady(health map[string]interface{}) bool {
	if running, ok := health["running"].(bool); ok && !running {
		return false
	}
	if status, ok := health["status"].(string); ok {
		return status == "ok"
	}
	return false
}

func nestedInt(values map[string]interface{}, section, key string) int {
	nested, ok := values[section].(map[string]interface{})
	if !ok {
		return 0
	}
	switch value := nested[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	default:
		return 0
	}
}
