package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/suite"
)

type HTTPInferenceStreamTestSuite struct {
	suite.Suite
	recorder *httptest.ResponseRecorder
	ginCtx   *gin.Context
	ctx      context.Context
	cancel   context.CancelFunc
	stream   *httpInferenceStream
}

func (s *HTTPInferenceStreamTestSuite) SetupTest() {
	gin.SetMode(gin.TestMode)
	s.recorder = httptest.NewRecorder()
	s.ginCtx, _ = gin.CreateTestContext(s.recorder)
	s.ctx, s.cancel = context.WithCancel(context.Background())
	s.stream = &httpInferenceStream{
		ctx:    s.ctx,
		writer: s.ginCtx.Writer,
	}
}

func (s *HTTPInferenceStreamTestSuite) TearDownTest() {
	if s.cancel != nil {
		s.cancel()
	}
}

func (s *HTTPInferenceStreamTestSuite) TestHTTPInferenceStreamSerializesEventsAndHonorsCancel() {
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		errs <- s.stream.writeEvent("token", map[string]string{"token": "hello"})
	}()
	go func() {
		defer wg.Done()
		errs <- s.stream.writeComment("keepalive")
	}()
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			s.Failf("write failed", "%v", err)
		}
	}

	if s.recorder.Code != http.StatusOK {
		s.Failf("expected stream to start with 200", "got %d", s.recorder.Code)
	}
	body := s.recorder.Body.String()
	if !strings.Contains(body, "event: token") {
		s.Failf("missing token event", "%q", body)
	}
	if !strings.Contains(body, ":keepalive\n\n") {
		s.Failf("missing keepalive comment", "%q", body)
	}

	s.cancel()
	if err := s.stream.writeEvent("token", map[string]string{"token": "late"}); err == nil {
		s.Fail("expected canceled context error")
	}
}

func TestHTTPInferenceStreamTestSuite(t *testing.T) {
	suite.Run(t, new(HTTPInferenceStreamTestSuite))
}
