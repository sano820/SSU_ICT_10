# SSU_ICT_10

```
/multi-agent-project
├── agents/             # 핵심 Agent 관련 코드
│   ├── base_agent.py   # 모든 Agent가 공통으로 상속받는 기본 클래스
│   ├── planning_agent.py
│   ├── execution_agent.py
│   └── ...
├── experiments/        
│   ├── sangsun/       
│   ├── youngjun/          
│   │   └── ...
│   └── ...
├── src/                # 실험이 완료되어 통합된 코드
│   ├── main.py         # 프로젝트의 메인 실행 파일
│   ├── components/
│   │   ├── tools.py
│   │   ├── memory.py
│   │   └── ...
├── docs/               # 문서화 폴더
│   ├── README.md       # 프로젝트 설명
│   └── design_spec.md
├── requirements.txt
├── .gitignore
└── tests/       
```
