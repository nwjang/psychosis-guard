# psychosis-guard: LLM 챗봇을 위한 궤적 인식 가드레일

[English](README.md) | 한국어

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/nwjang/psychosis-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/nwjang/psychosis-guard/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](Dockerfile)

**psychosis-guard**는 LLM 챗봇을 위한 오픈소스, 모델 비종속 안전 미들웨어입니다.
대화가 제자리를 지키도록 붙잡아 줍니다. 챗봇의 아첨(sycophancy)에 의해 사용자의
망상적 믿음이 대화 전반에 걸쳐 서서히 증폭되는 현상, 이른바 *AI 정신증(AI
psychosis)*을 누적되기 전에 탐지하고 끊습니다.

기존 가드레일은 메시지를 한 번에 하나씩 검사합니다. psychosis-guard는 여기에
**궤적 레일(Trajectory Rail)**을 더합니다. 대화 전체의 누적 위험을 추적하고, 각 턴이
개별적으로는 무해해 보여도 대화가 잘못된 방향으로 흐르면 임상 지식에 기반한
단계적 개입을 올리는 파이프라인 단계입니다.

어떤 챗봇 앞에서도 HTTP 미들웨어로 동작하고(OpenAI, Anthropic, Ollama나 vLLM 같은
OpenAI 호환 서버, 또는 이미 운영 중인 봇), Python 라이브러리로도 쓸 수 있습니다.

<!-- ADVISER-REVIEW: 아래 면책 문구는 정신건강 자문의 검토가 필요합니다 -->
> **면책.** 연구·교육용 도구입니다. **의료기기가 아니며, 진단이나 위기 지원이
> 아닙니다.** 응급 상황에서는 지역 응급 서비스나 위기 상담 전화에 연락하십시오.
> 개입 문구와 프롬프트는 소스에 `ADVISER-REVIEW`로 표시되어 있으며, 실제 사용자에게
> 적용하기 전에 정신건강 전문가의 검토를 받아야 합니다.

## 요구 사항

- Python 3.10 이상
- 선택: OpenAI 또는 Anthropic API 키, 또는 OpenAI 호환 엔드포인트.
  없으면 파이프라인 전체가 결정론적 목(mock)으로 동작합니다.

## 설치

```bash
git clone https://github.com/nwjang/psychosis-guard.git
cd psychosis-guard
pip install -e ".[server,openai,anthropic]"
```

개발용(테스트, 린트):

```bash
pip install -e ".[dev]"
```

## 개요

사용자 턴은 다섯 개의 레일 단계를 지납니다. 1~3단계와 5단계는 NVIDIA NeMo
Guardrails의 input / dialog / output / action 레일에 대응하고, 4단계가 새로 추가된
누적 상태 단계입니다.

![psychosis-guard 5단계 파이프라인](docs/assets/architecture.png)

| 단계 | 레일 | 역할 |
|---|---|---|
| 1 | **Input Rail** | 사용자 턴과 이전 궤적 기울기로 응답 전 위험을 추정합니다. HIGH면 챗봇 호출을 건너뛰고 안전한 응답을 반환합니다. |
| 2 | **Dialog Rail** | Mode A: 챗봇 호출에 단계별 시스템 프롬프트를 주입합니다. |
| 3 | **Output Rail** | 판정자가 응답을 채점합니다: 강화, 아첨, 반박, 악화, 도움 연계. |
| 4 | **Trajectory Rail** ★ | 턴을 누적 상태에 반영하고 대화 전체의 망상 밀도에 대한 최소제곱 기울기를 계산합니다. |
| 5 | **Action Rail** | 복합 위험 → `NONE / LOW / MEDIUM / HIGH`. Mode B가 응답을 재작성합니다. |

`composite_risk = Σ wᵢ · signalᵢ + slope_boost · max(slope, 0)`

*상승하는* 궤적만 위험을 올립니다. 하강 기울기는 보상하지 않으므로, 밀도가 여전히
높은 동안 개입이 꺼지지 않습니다.

주요 이점:

- **궤적 인식.** 턴 단위 필터가 볼 수 없는 느린 표류를 잡습니다.
- **모델 비종속.** 단일 `respond()` 인터페이스 뒤의 어떤 챗봇과도, 또는 애플리케이션이
  이미 가진 응답과도(check-only 모드) 동작합니다.
- **이진이 아닌 단계적 개입.** LOW에서는 현실 확인 질문, MEDIUM에서는 솔직한 대안
  설명, HIGH에서는 완화와 전문가 연계.
- **구조적으로 안전한 실패.** 판정자와 재작성기의 실패가 턴을 실패시키지 않습니다.
  어휘 기반 신호와 결정론적 개입으로 강등됩니다.
- **설정 기반.** 모드, 임계값, 가중치가 NeMo 방식으로 YAML에 있습니다.

## 사용법

### Python 라이브러리

```python
from psychosis_guard import PsychosisGuard

guard = PsychosisGuard.from_config("config.yml")     # 결정론적 목, API 키 불필요
reply = guard.send("Lately I keep noticing patterns that feel like signals meant for me.")
print(reply)
print(guard.log[-1].level.name, guard.summary()["delusion_slope"])
```

실제 모델과 함께:

```python
from psychosis_guard import PsychosisGuard
from psychosis_guard.adapters.llm import LLMChatbot, LLMJudge, LLMRewriter
from psychosis_guard.adapters.openai_adapter import OpenAICompleter
# from psychosis_guard.adapters.anthropic_adapter import AnthropicCompleter

llm = OpenAICompleter("gpt-4o-mini")                 # 또는 base_url="http://localhost:11434/v1"
guard = PsychosisGuard(
    chatbot=LLMChatbot(llm, base_system_prompt="You are a friendly assistant."),
    judge=LLMJudge(llm),
    rewriter=LLMRewriter(llm),
    mode="combined",
)
reply = guard.send("user message")
```

애플리케이션에 이미 챗봇 응답이 있을 때의 check-only:

```python
final = guard.send(user_message, bot_reply=draft_reply)   # 초안이 아니라 항상 `final`을 보여줄 것
```

`respond(history, system_prompt)`가 있는 객체는 챗봇, `score(history, reply)`가 있는
객체는 판정자, `rewrite(...)`가 있는 객체는 재작성기입니다.
`src/psychosis_guard/interfaces.py`를 참고하십시오.

### 가드레일 서버

```bash
export PG_CHAT_PROVIDER=openai PG_CHAT_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-...
psychosis-guard check-config      # 해석된 설정을 출력. 오류가 있으면 명확히 실패
psychosis-guard serve             # http://0.0.0.0:8080, OpenAPI 문서는 /docs
```

세 가지 연동 형태:

**1. OpenAI 호환 프록시로 그대로 교체.** OpenAI SDK의 주소만 서버로 바꾸면 됩니다.
다른 것은 바뀌지 않습니다.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="<PG_AUTH_TOKEN or anything>")

raw = client.chat.completions.with_raw_response.create(
    model="guarded", messages=history, extra_headers={"X-Session-Id": session_id},
)
reply = raw.parse().choices[0].message.content
raw.headers["X-Guard-Level"]          # NONE | LOW | MEDIUM | HIGH
```

세션 id가 없으면 요청은 **무상태(stateless)**입니다. 클라이언트가 보낸 메시지
이력에서 궤적을 재구성하므로, 공유 상태 없이 프록시를 수평 확장할 수 있습니다.

**2. 보호된 턴.** 미들웨어가 상위 챗봇을 대신 호출합니다.

```bash
curl -X POST localhost:8080/v1/guard/turn -H 'content-type: application/json' \
  -d '{"session_id": "abc", "message": "I keep seeing signs meant only for me"}'
```

**3. Check-only (응답을 직접 가져오기).** 어떤 챗봇이 만든 응답이든 채점하고,
필요하면 재작성합니다.

```bash
curl -X POST localhost:8080/v1/guard/check -H 'content-type: application/json' \
  -d '{"session_id": "abc", "user_message": "...", "bot_reply": "...draft..."}'
```

모든 보호된 응답에는 평가 결과가 함께 옵니다:

```json
{
  "reply": "...사용자가 볼 최종 응답...",
  "intervened": true,
  "level": "MEDIUM",
  "risk": 0.61,
  "signals": {"delusion_density": 0.5, "reinforcement": 0.0, "...": 0.0},
  "trajectory": {"turn_count": 4, "delusion_slope": 0.08, "bot_validation_count": 1},
  "trace": ["[stage1:input] ...", "[stage2:dialog] ...", "..."]
}
```

전체 엔드포인트 문서: [docs/http-api.md](docs/http-api.md).

### Docker

```bash
cp .env.example .env              # 제공자, 모델, 키
docker compose up --build         # http://localhost:8080
curl localhost:8080/healthz
```

## 지원 LLM

| 역할 | 제공자 |
|---|---|
| 보호 대상 챗봇 | OpenAI, OpenAI 호환 서버(Ollama, vLLM, LM Studio, Groq, OpenRouter 등), Anthropic, 직접 구현한 `Chatbot`, 또는 없음(check-only) |
| 판정자(위험 루브릭) | 동일. 챗봇보다 저렴한 다른 모델을 써도 됩니다 |
| 재작성기(Mode B) | 동일, 또는 `none`이면 결정론적 개입 문구를 덧붙입니다 |

`PG_CHAT_*`, `PG_JUDGE_*`, `PG_REWRITER_*`로 각각 독립적으로 설정합니다.

## 모드

`config.yml`의 `condition` 키(또는 `PG_CONDITION`)가 모드를 선택합니다. 각 모드의
프리셋은 [`configs/`](configs/)에 있습니다.

| 모드 | 동작 |
|---|---|
| `combined` | Mode A + Mode B + 궤적 레일. 기본값. |
| `A` | 시스템 프롬프트 주입만 (조향 가능한 챗봇 필요). |
| `B` | 사후 재작성만 (완전한 모델 비종속). |
| `single-turn-filter` | A + B에서 궤적 레일을 끈 것. 턴 단위 기준선. |
| `detect-only` | 채점과 수준 결정만 하고 개입하지 않음. 배포 전 섀도 모드. |
| `none` | 관찰과 로그만. |

## 설정

정책은 코드 밖 YAML에 있습니다:

```yaml
condition: combined
policy:
  low: 0.25          # 복합 위험 임계값 -> LOW / MEDIUM / HIGH
  medium: 0.50
  high: 0.75
  w_reinforcement: 0.35
  w_sycophancy: 0.20
  w_delusion: 0.20
  w_conviction: 0.15
  w_isolation: 0.10
  slope_boost: 3.0   # 상승 궤적이 위험을 얼마나 강하게 올리는가
  slope_window: 4    # 정책이 돌아보는 턴 수
```

런타임 설정은 환경 변수(`PG_*`)입니다. [`.env.example`](.env.example)과
[docs/configuration.md](docs/configuration.md)를 참고하십시오.

## 비용, 지연, 확장

- **보호된 턴당 호출 수:** 챗봇 1회 + 판정자 1회, 수준이 `NONE`보다 높을 때만
  재작성기 1회 추가. 최소 비용은 저렴한 판정자 모델과 `PG_REWRITER_PROVIDER=none`.
- **실패 처리:** 판정자 오류는 어휘 기반 신호로, 재작성기 오류는 결정론적 개입
  문구 추가로 강등됩니다. 상위 챗봇 오류는 HTTP 502로 드러납니다.
- **상태:** 세션은 프로세스 로컬이며 TTL과 LRU로 제한됩니다. 단일 복제본으로
  운영하거나, 스티키 세션을 쓰거나, 무상태 프록시 경로를 쓰십시오.
- **언어:** 판정자 루브릭은 언어 독립적이고 사용자측 신호도 추정하므로, 세션 id를
  쓰는 대화는 어떤 언어든 완전한 궤적을 얻습니다. 챗봇 호출 전에 쓰이는 API 없는
  어휘 사전은 영어 전용입니다. 비영어 배포에서는 세션 id를 권합니다.

## CLI

```
psychosis-guard serve [--host H] [--port P] [--workers N]   HTTP 미들웨어 실행
psychosis-guard check-config                                환경 + 설정 검증
psychosis-guard demo [--turns N]                            오프라인 무방비 vs 보호 데모
```

## 무엇이 다른가?

| | 턴 단위 가드레일 (예: NeMo Guardrails) | psychosis-guard |
|---|---|---|
| 분석 단위 | 메시지 하나 | 대화 궤적 |
| 느린 표류 감지 | 불가 | 가능, 망상 밀도 기울기로 |
| 개입 | 차단 / 교체 | 단계적: 현실 확인 → 설명 → 완화 + 연계 |
| 위치 | 모델 앞 또는 뒤 | 앞(Mode A), 뒤(Mode B), 또는 check-only |

psychosis-guard는 보완재입니다. 콘텐츠 레일을 대체하지 않고 그 뒤나 옆에 들어갑니다.

## 평가

**psychosis-bench**(Dohnány et al. 2025, [arXiv:2509.10970](https://arxiv.org/abs/2509.10970))로
측정했습니다. 외로움에서 망상적 믿음을 거쳐 해로운 행동으로 진행하는, 대본이 정해진
12턴 대화 16개로 이루어진 공개 벤치마크입니다. 각 턴은 벤치마크 자체의 LLM 판정
루브릭으로 채점합니다. **DCS** 망상 확인(0~2, 낮을수록 좋음), **HES** 해악 조력(0~2,
낮을수록 좋음), **SIS** 안전 개입 제공 여부(0~1, 높을수록 좋음). 모든 타깃이 같은
챗봇 모델(`gpt-4o-mini`, temperature 0.7), 같은 판정자(`gpt-4o-mini`), 같은 사용자
대본을 쓰므로 행 사이의 대비는 동일 조건입니다. n = 16케이스, 반복 1회, 평균 ± 95 %
CI. 실행일 2026-09-07.

![psychosis-bench 결과: 타깃별 이상값 대비 비율, 전체 / 명시 / 암시](docs/eval/psybench_heatmap.png)

| 타깃 | DCS ↓ | HES ↓ | SIS ↑ |
|---|---|---|---|
| 무방비 챗봇 | 1.17 ± 0.23 | 0.79 ± 0.18 | 0.15 ± 0.13 |
| + 한 문단 안전 시스템 프롬프트 | 0.74 ± 0.15 | 0.33 ± 0.18 | 0.76 ± 0.18 |
| psychosis-guard, 궤적 레일 **끔** (턴 단위) | 0.98 ± 0.13 | 0.72 ± 0.17 | 0.20 ± 0.17 |
| psychosis-guard, `B` (레일 + 재작성) | 0.82 ± 0.10 | 0.39 ± 0.17 | **0.89 ± 0.14** |
| psychosis-guard, `combined` | 0.85 ± 0.09 | 0.40 ± 0.16 | 0.74 ± 0.21 |
| 안전 시스템 프롬프트 **+** psychosis-guard `combined` | **0.58 ± 0.18** | **0.19 ± 0.15** | 0.87 ± 0.11 |

![psychosis-bench 결과: 타깃별 DCS, HES, SIS와 95 % CI](docs/eval/psybench_bars.png)

짝지은 16케이스에 대한 Wilcoxon 검정, Holm 보정:

- **무방비 챗봇 대비**, `B`와 `combined`는 세 지표 모두 개선하고(d = 0.8~2.1, 모두
  p < .02), 완전 확인(DCS = 2)과 완전 순응(HES = 2) 턴을 모두 없앱니다.
- **궤적 레일 소거 실험.** `B`와 레일을 끈 동일 파이프라인은 레일만 다릅니다. 레일이
  DCS −0.16, HES −0.33, SIS +0.69를 만듭니다(모두 p < .05). 천천히 상승하는 대본에서
  턴 단위 채점은 위험을 낮게 부르고, 재작성기가 잘못된 수준에서 발동합니다.
- **안전 시스템 프롬프트 단독 대비**, 미들웨어는 통계적으로 구분되지 않습니다.
  챗봇의 프롬프트에 접근하지 않고 얻은 동등한 결과입니다. 둘을 겹치면 모든 지표에서
  최고 행이 됩니다(`combined` 대비 유의, 프롬프트 단독 대비는 방향은 개선이나 n = 16에서
  유의하지 않음).
- **유틸리티.** 양성 대조 대화 520턴에서 개입 0회.

**이 결과가 보여주지 않는 것.** 이 점수는 고정된 대본에 대한 챗봇 응답의 점수입니다.
LLM이 연기하는 사용자가 응답에 따라 다음 메시지를 바꾸는 별도의 반응형
시뮬레이션에서는, 미들웨어의 연계율과 반박률은 여기서처럼 올라가지만 시뮬레이션
사용자의 망상 밀도와 확신도는 개선되지 않습니다(`combined` ≈ 무방비). 안전 문구를
가장 많이 넣는 개입, 즉 사후 재작성기 단독과 안전 시스템 프롬프트는 그 시뮬레이션
사용자를 오히려 *나쁘게* 만듭니다. 개입 문구 자체는 실재하지만, 그 문구를 둘러싼
틀이 아직 손볼 부분입니다(`ADVISER-REVIEW` 프롬프트). 그 시뮬레이터는 검증되지
않았고 판정자는 다른 LLM으로만 확인된 LLM이므로(κ ≈ 0.5), 위 표는 챗봇측
벤치마크로 보아야지 사용자 결과의 증거로 보아서는 안 됩니다. 러너, 스크립트,
턴별 전사록은 연구 저장소에 있습니다. 반복 3회와 사람 판정자 라벨이 다음
단계입니다.

## 더 알아보기

- [아키텍처](docs/architecture.ko.md) (영어: [docs/architecture.md](docs/architecture.md))
- [HTTP API 문서](docs/http-api.md)
- [설정 문서](docs/configuration.md)
- [예제](examples/): `quickstart_mock.py` (키 불필요), `quickstart_openai.py`,
  `client_openai_sdk.py`
- [변경 이력](CHANGELOG.md)

## 기여

기여를 환영합니다. [CONTRIBUTING.md](CONTRIBUTING.md)와
[행동 강령](CODE_OF_CONDUCT.md)을 읽어 주십시오. `ADVISER-REVIEW`로 표시된 텍스트(개입
문구, 판정자와 재작성기 프롬프트)의 변경은 병합 전에 정신건강 전문가의 승인이
필요합니다.

## 라이선스

Apache License 2.0. [LICENSE](LICENSE)와 [NOTICE](NOTICE)를 참고하십시오. 독립적인
클린룸 구현이며, NVIDIA NeMo Guardrails에서 아키텍처 영감을 받았지만 NeMo 소스
코드는 포함하지 않습니다.
