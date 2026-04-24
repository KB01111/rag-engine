package api

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

const (
	apiErrorInvalidRequest     = "invalid_request"
	apiErrorNotFound           = "not_found"
	apiErrorBackendUnavailable = "backend_unavailable"
	apiErrorInternal           = "internal_error"
)

type apiErrorBody struct {
	Error apiError `json:"error"`
}

type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func writeAPIError(c *gin.Context, status int, code, message string) {
	if code == "" {
		code = apiErrorInternal
	}
	if message == "" {
		message = http.StatusText(status)
	}
	c.JSON(status, apiErrorBody{
		Error: apiError{
			Code:    code,
			Message: message,
		},
	})
}

func backendError(c *gin.Context, err error) {
	if err == nil {
		writeAPIError(c, http.StatusInternalServerError, apiErrorInternal, "internal server error")
		return
	}
	writeAPIError(c, http.StatusBadGateway, apiErrorBackendUnavailable, err.Error())
}
