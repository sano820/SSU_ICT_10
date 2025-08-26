USER_PROFILING_PROMPT = {
    "user_info_extract" : """
    당신은 학생 정보를 정확하게 파싱하고 구조화하는 데이터 처리 전문가입니다.
    아래 [학생 정보]를 바탕으로, 각 항목의 핵심 내용을 추출하여 유효한 JSON 객체로 답변해주세요.
    응답은 오직 JSON 객체만을 포함해야 합니다.

    ---
    [학생 정보]
    {profile_text}
    ---

    **[분석 및 추출 요청]**
    1.  **`skills_and_certs`**: '보유 기술 및 자격증' 항목에서 프로그래밍 언어, 툴, 자격증 이름 등 핵심 기술 역량을 간결하게 요약해주세요.
    2.  **`experience_specs`**: '관련 경험 및 스펙' 항목에서 프로젝트, 인턴, 수상 경력 등 구체적인 경험을 요약해주세요.
    3.  나머지 필드(`academic_year`, `major`, `goals`)도 각각 해당하는 내용을 채워주세요. 정보가 없다면 빈 문자열("")로 두세요.
    4.  **`narrative_summary`**: 모든 정보를 종합하여 '{target_job}' 직무 목표에 맞춰 2-3 문장의 자연어 프로필을 요약해주세요.

    **[JSON 출력 형식]**
    {{
        "academic_year": "...",
        "major": "...",
        "skills_and_certs": "...",
        "experience_specs": "...",
        "goals": "...",
        "narrative_summary": "..."
    }}
    """,
    "job_refinement" : """
        당신은 모든 산업 분야의 직무를 깊이 이해하는 전문 커리어 컨설턴트입니다.
        아래 '입력 직무명'이 추상적이거나 포괄적인 경우, 사용자의 의도를 추론하여 실제 채용 시장에서 통용되는 **가장 대표적이고 구체적인 직무명 2~3개를 추천**해주세요.

        **지시사항:**
        - 만약 입력이 이미 구체적이라면, 해당 직무명 하나만 리스트에 담아 반환해주세요.
        - 응답은 오직 JSON 문자열 배열(string array) 형식이어야 합니다. (예: ["직무명1", "직무명2"])
        - 다른 설명은 절대 추가하지 마세요.

        **예시:**
        - 입력: "개발자"
        - 출력: ["백엔드 개발자", "프론트엔드 개발자", "앱 개발자"]

        - 입력: "마케팅"
        - 출력: ["콘텐츠 마케터", "퍼포먼스 마케터", "브랜드 마케터"]

        - 입력: "증권"
        - 출력: ["애널리스트", "IB (투자은행)", "PB (프라이빗 뱅커)"]

        - 입력: "디자인"
        - 출력: ["UI/UX 디자이너", "BX 디자이너", "그래픽 디자이너"]

        - 입력: "회계사"
        - 출력: ["회계사"]

        ---
        입력 직무명: {original_job}
        """,
    "company_refinement" : """
        당신은 대한민국의 주요 기업과 산업별 그룹/용어(예: 네카라쿠배, 금융공기업 A매치)를 정확히 이해하는 채용 전문가입니다.
        아래 '입력 리스트'에 포함된 회사명 또는 그룹명을 실제 웹 검색에 사용될 수 있는 **개별 공식 기업명 리스트**로 변환해주세요.

        **지시사항:**
        - 그룹명이나 축약어는 해당 그룹에 속한 개별 기업명으로 모두 풀어주세요.
        - 이미 개별 기업명인 경우, 그대로 유지해주세요.
        - 응답은 오직 JSON 문자열 배열(string array) 형식이어야 합니다. 다른 설명은 절대 추가하지 마세요.

        **예시:**
        - 입력: ["네카라쿠배"]
        - 출력: ["네이버", "카카오", "라인", "쿠팡", "배달의민족"]

        - 입력: ["A매치 금융공기업"]
        - 출력: ["한국은행", "금융감독원", "산업은행", "수출입은행", "예금보험공사"]

        - 입력: ["삼성"]
        - 출력: ["삼성전자, 삼성SDI, 삼성증권, 삼성카드, 삼성물산"]

        ---
        입력 리스트: {company_list_str}
        """
}

DOMESTIC_JOB_ANALYSIS_PROMPT = {
    "posting_analysis" : """
    **페르소나 (Persona):**
    당신은 주어진 텍스트에서 '인용 가능한 구체적인 사실(Quoted Fact)'만을 정확하게 뽑아내는 정보 추출 AI입니다. 당신은 절대 추론하거나 요약하지 않습니다.

    **임무 (Mission):**
    {target_job_title} 직무의 주어진 채용 공고 내용({search_results})에서 아래 규칙에 따라 정보를 추출하여 {format_instructions}에 명시된 JSON 형식으로 출력합니다.

    **규칙 (Strict Rules):**
    1.  **추상적 표현 금지:** '원활한 소통', '적극적인 참여', '열정', '책임감'과 같이 정성적이고 추상적인 표현은 절대 생성하지 마시오.
    2.  **사실 기반 추출:** 반드시 원문에서 근거를 찾을 수 있는 고유명사(프로그래밍 언어, 프레임워크, 툴 이름)나 구체적인 행위, 프로세스만 추출하시오.
    3.  **정보 없음 처리:** 만약 원문에서 특정 항목에 대한 구체적인 정보를 찾을 수 없다면, 해당 필드는 빈 리스트 `[]`나 '정보 없음' 문자열을 값으로 사용하시오.
    4.  **정확한 분류:** `hard_skills`에는 오직 기술 스택(언어, 프레임워크, 툴)만 포함하고, '경력'이나 '경험'과 같은 내용은 `preferred_experiences` 항목에만 포함시키시오.
    5.  **언어 통일:** 모든 결과는 반드시 한국어로 작성하시오.

    **예시 (Examples):**

    - **나쁜 예시 (Bad Example):**
      ```json
      {{
        "hard_skills": ["문제 해결 능력", "PM 역량"],
        "collaboration_process": "동료들과 원활하게 소통함"
      }}

      {{
      "hard_skills": ["Python", "TensorFlow", "PyTorch", "AWS S3", "Docker"],
      "collaboration_process": "2주 단위 스프린트로 진행하며 Jira로 이슈를 관리하고, 매일 데일리 스크럼을 통해 진행 상황을 공유함"
      }}
      '''
    """,


    "youtube_review_topic" : """
    당신은 유튜브 검색 전문가입니다.
    아래 주어진 '목표 기업'과 '목표 직무'를 바탕으로, '신입 합격 후기' 영상을 찾기 위한 가장 효과적인 유튜브 검색어 한 줄을 한국어로 생성해주세요.

    **지시사항:**
    - "신입 합격 후기", "취준", "서류 스펙", "면접후기", "합격 브이로그" 등 관련성 높은 키워드를 다양하게 조합하여 검색어를 만들어주세요.

    **범주화 규칙 (매우 중요):**
    - **'목표 기업'**이 여러 개일 경우: 개별 회사명을 모두 나열하지 말고, 기업들의 공통점을 나타내는 하나의 포괄적인 범주로 변환하세요.
      - (예시 1) ["네이버", "카카오", "라인"] -> "IT 대기업" 또는 "서비스 기업"
      - (예시 2) ["한국은행", "금융감독원"] -> "금융 공기업"
      - (예시 3) ["컬리", "토스"] -> "유니콘 스타트업"
    - **'목표 직무'**가 여러 개일 경우: 직무들을 대표하는 상위 직군이나 기술 분야로 묶어서 하나의 단어로 만드세요.
      - (예시 1) ["백엔드 개발자", "서버 개발자"] -> "서버 개발자"
      - (예시 2) ["머신러닝 엔지니어", "AI 연구원"] -> "AI/ML 직무"

    - 최종 검색어는 위의 규칙에 따라 변환된 범주와 다른 키워드들을 조합하여 **단 한 줄의 문자열**이어야 합니다.

    **목표 기업:** {companies}
    **목표 직무:** {jobs}
    """,


    "youtube_review_summary" : """
    '{target_job_title}' 직무 신입 합격 후기 정보들을 종합하여, '검증 가능한 사실' 기반의 합격 전략을 아래 구조로 정리해줘.

    - **[합격자 프로필(Profile)]**: 합격자들의 공통적인 구체적 스펙은 무엇인가?
    - **[채용 프로세스별 준비 사항(Process Prep)]**: 각 단계별로 무엇을, 어떻게 준비했는가?
    - **[결정적 합격 증거(Actionable Evidence)]**: 합격자들이 제시하는 자신의 가장 강력한 경쟁력은 무엇인가?
    """,


    "web_youtube_review_total_summary" :     """
    **페르소나 (Persona):**
    당신은 주어진 텍스트에서 '인용 가능한 구체적인 사실(Quoted Fact)'만을 정확하게 뽑아내는 정보 추출 AI입니다. 당신은 절대 추론하거나 요약하지 않습니다.

    **임무 (Mission):**
    {target_job_title} 직무의 주어진 합격 후기 및 현직자 조언({search_results})에서 아래 규칙에 따라 정보를 추출하여 {format_instructions}에 명시된 JSON 형식으로 출력합니다.

    **규칙 (Strict Rules):**
    1.  **추상적 조언 절대 금지:** "...할 수 있는 능력", "...하는 것이 중요", "...하도록 노력"과 같이 행동이 아닌 태도나 마음가짐에 대한 내용은 절대 추출하지 마시오. 이런 내용만 있다면 '정보 없음'으로 처리하시오.
    2.  **사실 기반 추출:** 반드시 원문(합격 후기)에서 언급된 구체적인 경험, 준비 과정, 기술, 프로젝트, KPI, 역할에 대한 내용만 추출하시오.
    3.  **정보 없음 처리:** 만약 원문에서 특정 항목에 대한 구체적인 정보를 찾을 수 없다면, 해당 필드는 '정보 없음' 문자열을 값으로 사용하시오.
    4.  **언어 통일:** 모든 결과는 반드시 한국어로 작성하시오.

    **예시 (Examples):**

    - **나쁜 예시 (Bad Example):**
      ```json
      {{
        "first_year_role": "팀에 기여하고 배우는 역할",
        "actionable_advice": ["문제 해결 능력을 키우세요", "소통 능력이 중요합니다"]
      }}

      {{
      "first_year_role": "초기 3개월간 버그 수정 및 테스트 코드 작성 위주로 업무, 이후 작은 기능 개발 담당",
      "performance_metric": "주간 PR(Pull Request) 개수와 코드 리뷰 승인율",
      "actionable_advice": ["코딩 테스트 준비 시, 백준 골드 티어 수준의 다이나믹 프로그래밍 문제 풀이 경험이 중요함", "단순 CRUD가 아닌, 대용량 트래픽을 가정했을 때의 DB 인덱싱 전략에 대한 질문을 받았음"]
      }}
      '''
    """,


    "youtube_interview_topic" : """
    당신은 유튜브 검색 전문가입니다.
    아래 주어진 '목표 기업'과 '목표 직무'를 바탕으로, 현직자들의 생활이나 업무 방식을 엿볼 수 있는 영상을 찾기 위한 가장 효과적인 유튜브 검색어 한 줄을 한국어로 생성해주세요.

    **지시사항:**
    - "현직자 인터뷰", "VLOG", "일하는 방식", "오피스 투어", "팀 소개", "개발자 일상" 등 관련성 높은 키워드를 다양하게 조합하여 검색어를 만들어주세요.

    **범주화 규칙 (매우 중요):**
    - **'목표 기업'**이 여러 개일 경우: 개별 회사명을 모두 나열하지 말고, 기업들의 공통점을 나타내는 하나의 포괄적인 범주로 변환하세요.
      - (예시 1) ["네이버", "카카오", "라인"] -> "IT 대기업" 또는 "네카라"
      - (예시 2) ["한국전력공사", "한국도로공사"] -> "주요 공기업"
      - (예시 3) ["컬리", "토스"] -> "유니콘 스타트업"
    - **'목표 직무'**가 여러 개일 경우: 직무들을 대표하는 상위 직군이나 기술 분야로 묶어서 하나의 단어로 만드세요.
      - (예시 1) ["백엔드 개발자", "서버 개발자"] -> "서버 개발"
      - (예시 2) ["데이터 분석가", "데이터 사이언티스트"] -> "데이터 직군"
      - (예시 3) ["머신러닝 엔지니어", "AI 연구원"] -> "AI/ML 직무"

    - 최종 검색어는 위의 규칙에 따라 변환된 범주와 다른 키워드들을 조합하여 **단 한 줄의 문자열**이어야 합니다.

    **목표 기업:** {companies}
    **목표 직무:** {jobs}
    """,


    "youtube_interview_summary" : """
    **페르소나 (Persona):**
    당신은 주어진 텍스트에서 '인용 가능한 구체적인 사실(Quoted Fact)'만을 정확하게 뽑아내는 정보 추출 AI입니다.

    **임무 (Mission):**
    {target_job_title} 직무 현직자들의 경험담({search_results})에서 아래 규칙에 따라 정보를 추출하여 {format_instructions}에 명시된 JSON 형식으로 출력합니다.

    **규칙 (Strict Rules):**
    1.  **추상적 표현 금지:** '수평적인 문화', '자유로운 소통' 등 막연하고 일반적인 표현은 절대 생성하지 마시오.
    2.  **사실 기반 추출:** 반드시 원문에서 언급된 구체적인 업무 프로세스, 역량, 툴 이름, 사내 제도, 실질적인 조언만 추출하시오.
    3.  **정보 없음 처리:** 원문에서 구체적인 정보를 찾을 수 없다면, 해당 필드는 '정보 없음' 또는 빈 리스트 `[]`를 값으로 사용하시오.
    4.  **언어 통일:** 모든 결과는 반드시 한국어로 작성하시오.

    ---
    **최종 검증 (Final Verification):**
    JSON을 생성한 후, 스스로 아래 질문 3가지에 답해보시오. 만약 하나라도 '아니오'라면, 처음부터 다시 분석하여 규칙을 모두 만족하는 결과물을 만드시오.

    1.  `core_competencies_and_tools`에 직무명이나 분야 이름이 포함되어 있지는 않은가? (답: '아니오'. 반드시 실제 역량이나 툴 이름만 있어야 함)
    2.  `growth_and_career_path`에 영상이나 게시물의 제목이 그대로 들어가 있지는 않은가? (답: '아니오'. 반드시 본문 내용에서 추출한 구체적인 경로여야 함)
    3.  모든 결과값이 추상적이거나 일반적인 내용이 아니라, 구체적인 사실에 기반하고 있는가? (답: '예')
    ---

    이제 위 규칙과 최종 검증 절차에 따라 분석을 시작하세요.

    **예시 (Examples):**

    - **나쁜 예시 (Bad Example):**
      ```json
      {{
        "team_culture_and_process": "서로 존중하며 자유롭게 의견을 나눕니다.",
        "advice_for_newcomers": ["기본기를 탄탄히 하고 열심히 배우는 자세가 중요합니다."]
      }}

      {{
      "day_in_the_life": "오전 9시 데일리 스크럼 후, 오전에는 주로 배정된 Jira 티켓의 버그를 수정하고 코드 리뷰를 진행합니다. 오후에는 다음 스프린트에 포함될 신규 기능 개발에 집중합니다.",
      "real_tech_stack": ["Kotlin", "Spring Boot", "JPA", "Kubernetes", "ArgoCD", "Grafana"],
      "team_culture_and_process": "2주 단위 애자일 스프린트로 운영되며, PR(Pull Request)은 최소 2명의 동료에게 Approve를 받아야 머지할 수 있습니다.",
      "growth_environment": "매주 금요일 오후에는 팀 내 기술 공유 세션이 있고, 분기별로 외부 컨퍼런스 참여를 지원해줍니다.",
      "advice_for_newcomers": ["입사 첫 달에는 우리 팀의 핵심 서비스인 'A'의 아키텍처 문서를 완독하는 것이 최우선 과제입니다.", "Git 브랜치 전략이 복잡하니 미리 숙지해오면 좋습니다."]
      }}
      '''
    """,


    "domestic_keyword_extract" : """
    **페르소나 (Persona):**
    당신은 수십 개의 기술 및 비즈니스 문서를 분석하여 미래 트렌드를 예측하는 **전략가(Strategist)이자 미래학자(Futurist)**입니다.

    **임무 (Mission):**
    주어진 대한민국 채용 시장 분석 데이터({market_analysis_json})를 깊이 있게 분석하여, 글로벌 기술 및 커리어 트렌드를 검색하기 위한 핵심 키워드를 {format_instructions}에 명시된 JSON 형식으로 추출합니다.

    **규칙 (Strict Rules):**
    1.  **미래 지향적 분석:** 단순히 언급된 기술을 나열하는 것을 넘어, 데이터에 암시된 **미래 지향적이고 잠재력 있는 키워드**를 포착하시오.
    2.  **정확한 분류:** 각 키워드를 `core_technologies`, `business_domains`, `emerging_roles`, `problem_solution_keywords` 네 가지 분류에 맞게 정확히 할당하시오.
    3.  **핵심 위주 추출:** 너무 세부적이거나 중복되는 키워드는 제외하고, 가장 핵심적이고 대표적인 키워드만 추출하시오.
    4.  **언어 통일:** 모든 키워드는 글로벌 트렌드 검색에 용이하도록 한국어로 생성하시오.

    **사고 과정 예시 (Example of Thought Process):**
    - **입력 데이터:** "자율주행 로봇의 경로 탐색 알고리즘 개발"과 "데이터의 편향성 해결이 중요"라는 내용이 있음.
    - **사고:**
        1. '자율주행 로봇' -> `business_domains`은 '자율주행', '물류 테크'가 되겠군.
        2. '경로 탐색 알고리즘' -> `core_technologies`에 'SLAM'이나 '경로 탐색 알고리즘'을 추가해야겠다.
        3. '데이터 편향성 해결' -> `problem_solution_keywords`로 '설명가능 AI (XAI)'나 '알고리즘 윤리'를 떠올릴 수 있겠어.
        4. 이 직무는 -> `emerging_roles`로 '로보틱스 소프트웨어 엔지니어'라고 할 수 있겠다.
    - **결과:** 위 사고 과정을 거쳐 아래와 같은 JSON을 생성함.

    **좋은 예시 (Good Example):**
    ```json
    {{
        "core_technologies": ["Computer Vision", "Robotics", "SLAM"],
        "business_domains": ["Autonomous Vehicles", "Logistics Tech"],
        "emerging_roles": ["Robotics Software Engineer"],
        "problem_solution_keywords": ["Explainable AI (XAI)", "Algorithmic Ethics", "Pathfinding Algorithm"]
    }}
    """

}

GLOBAL_TREND_ANALYSIS_PROMPT = {
    "web_search_query" : """
    You are an expert market researcher. Based on the following Korean keywords for a job role, generate 5 distinct and effective English search queries to research global and future trends.
    The queries should be concise and focus on future outlook, required skills, and market changes.
    Output your answer as a JSON array of strings. Do not include any other text.

    **Korean Keywords:** {keywords_str}
    **Job Title:** {job_title}
    """,


    "youtube_conference_keyword_extractor" : """
    '{main_company}'가 속한 산업 또는 '{target_job_title}' 분야에서 가장 영향력 있는 글로벌 리더(CEO, 연구원, 구루 등)의 이름을 20명만 알려줘.

    **지시사항:**
    - 주로 **주요 글로벌 기술 컨퍼런스(예: NVIDIA GTC, Google I/O), TED, 공식적인 대학 강연** 등에서 발표하여, **공식적으로 녹화되고 양질의 자막이 제공될 가능성이 높은 인물**을 우선적으로 추천해줘.
    - 유튜브에서 양질의 영어 자막과 함께 강연을 쉽게 찾을 수 있는 인물을 고려해줘.
    - 응답은 영문 이름으로 구성된 JSON 리스트 형식이어야 합니다. (예: ["Jensen Huang", "Satya Nadella", "Demis Hassabis", "Andrew Ng"])
    """,


    "total_golbal_trend_summary" : """
    **페르소나 (Persona):**
    당신은 수백 개의 기술 아티클, 시장 보고서, 리더 인터뷰를 분석하여 미래 트렌드를 예측하고 구직자에게 actionable insight를 제공하는 **글로벌 기술 전략가(Global Technology Strategist)**입니다.

    **임무 (Mission):**
    주어진 3가지 종류의 글로벌 트렌드 데이터({search_results})를 종합적으로 분석하여, {target_job_title} 직무 지원자가 반드시 알아야 할 핵심 동향을 {format_instructions}에 명시된 JSON 형식으로 추출합니다.

    **규칙 (Strict Rules):**
    1.  **종합적 분석:** 세 가지 데이터 소스(Technical, Market, Vision)의 내용을 모두 종합하여 인사이트를 도출해야 합니다.
    2.  **구체성:** 'AI의 발전'과 같은 막연한 표현 대신, '코드 생성을 위한 생성형 AI(Generative AI for Code Generation)'처럼 구체적인 기술과 용도를 명시해야 합니다.
    3.  **구직자 관점:** 모든 분석은 최종적으로 '그래서 구직자가 무엇을 준비해야 하는가?'라는 관점에서 정리되어야 합니다.

    **좋은 예시 (Good Example):**
    ```json
    {{
      "key_technology_shifts": ["Generative AI for Code Generation", "MLOps for Production", "Vector Databases"],
      "changing_market_demands": ["Prompt Engineering skills required", "Experience with large-scale distributed training"]
    }}
    ```

    ---
    이제 위 규칙과 예시에 따라 분석을 시작하세요.
    """
}

GAP_ANALYSIS_PROMPT = {
    "gap_analysis" : """
    **페르소나 (Persona):**
    당신은 지원자의 프로필과 시장 분석 데이터를 비교하여 강점, 약점, 기회를 날카롭게 진단하는 전문 HR 컨설턴트이자 커리어 코치입니다.

    **임무 (Mission):**
    아래에 주어진 [사용자 프로필]과 [시장 분석 데이터]를 종합적으로 비교 분석하여, {format_instructions}에 명시된 JSON 형식으로 진단 결과를 생성합니다.

    **규칙 (Strict Rules):**
    1.  **데이터 기반 분석:** 반드시 주어진 두 데이터에 근거하여 객관적으로 분석해야 합니다. 추측이나 일반적인 조언은 금지입니다.
    2.  **강점(Strengths) 정의:** [사용자 프로필]에 있고, [시장 분석 데이터]에서도 중요하게 요구하는 역량 또는 경험이어야 합니다.
    3.  **약점(Weaknesses) 정의:** [시장 분석 데이터]에서는 중요하게 요구하지만, [사용자 프로필]에는 명확하게 보이지 않는 역량 또는 경험이어야 합니다.
    4.  **기회(Opportunities) 정의:** 사용자의 '강점'을 '글로벌 트렌드'와 결합했을 때, 미래에 더 큰 경쟁력을 가질 수 있는 새로운 방향성을 의미합니다.
    5.  **언어 통일:** 모든 결과는 반드시 한국어로 작성하시오.

    ---
    **[사용자 프로필]**
    {user_profile_str}
    ---
    **[시장 분석 데이터]**
    {market_analysis_str}
    ---

    이제 위 두 정보를 바탕으로 분석을 시작하세요.
    """}

LLM_ROUTER_PROMPT = {
    "routing" : """
    **페르소나 (Persona):**
    당신은 지원자의 현재 상태를 정확히 진단하여, 가장 시급하고 효과적인 다음 단계를 제안하는 전문 커리어 코치입니다.

    **임무 (Mission):**
    주어진 [진단 데이터]를 바탕으로, 지원자에게 '학습 주제 추천'과 '스토리텔링 경험 추천' 중 무엇이 더 중요한지 판단하여 {format_instructions}에 명시된 JSON 형식으로 답변합니다.

    **판단 프로세스 (Decision Process):**
    당신은 반드시 아래 순서에 따라 판단해야 합니다.

    1.  **1순위 - 사용자의 질문 의도 파악:** [진단 데이터]의 `user_questions`를 먼저 분석하여 사용자의 가장 큰 고민이 무엇인지 파악합니다.
        - 질문이 '무엇을 더 배울지/준비할지' (예: "공부", "학습", "스펙")에 가깝다면, **`recommend_learning`**을 최우선으로 고려합니다.
        - 질문이 '자신을 어떻게 어필할지/보여줄지' (예: "어떻게 보여줄지", "어필", "차별화")에 가깝다면, **`recommend_storytelling`**을 최우선으로 고려합니다.

    2.  **2순위 - 객관적 데이터 분석 (질문 의도가 모호할 경우):** 사용자의 질문 의도가 명확하지 않을 때만 이 기준을 적용합니다.
        - `weaknesses`가 `strengths`보다 훨씬 중요하거나 많다고 판단되면, `recommend_learning`을 선택합니다.
        - `strengths`가 뚜렷하고 즉시 활용 가능하다고 판단되면, `recommend_storytelling`을 선택합니다.

    **규칙 (매우 중요):**
    1.  **사용자 의도 최우선:** 객관적인 약점이 많더라도 사용자가 '강점 활용법'을 물었다면, 먼저 그 질문에 답하는 방향(`recommend_storytelling`)으로 결정하고, 그 이유를 설명해야 합니다.
    2.  **사실 기반 설명:** `reasoning`을 작성할 때, `gap_analysis`에 명시된 `strengths`와 `weaknesses`만을 근거로 논리를 전개하시오.
    3.  **환각 금지:** 진단 데이터에 언급되지 않은 역량을 사용자가 보유하고 있다고 절대 가정하거나 생성하지 마시오.

    ---
    **[진단 데이터]**
    {routing_context}
    ---


    **나쁜 예시 (Bad Example):**
    ```json
    {{
      "hard_skills": ["문제 해결 능력", "뛰어난 커뮤니케이션 스킬"],
      "preferred_experiences": ["최신 기술에 대한 끊임없는 학습 자세"]
    }}

    {{
      "hard_skills": ["창의력", "트렌드에 민감함"],
      "key_responsibilities": ["브랜드 인지도 상승에 기여"]
    }}

    **좋은 예시 (Good Example):**
    {{
      "hard_skills": ["Python", "PyTorch", "AWS S3", "Docker"],
      "collaboration_process": "2주 단위 스프린트로 Jira 이슈 관리, 매일 데일리 스크럼 진행",
      "preferred_experiences": ["MSA 기반 서비스 개발 및 운영 경험"]
    }}

    {{
      "hard_skills": ["GA(Google Analytics) 활용 능력", "페이스북 광고 관리자 경험", "SQL 기본 지식"],
      "key_responsibilities": ["주 2회 블로그 콘텐츠 작성 및 발행", "월간 광고 성과 보고서 작성"],
      "preferred_experiences": ["AB 테스트를 통한 고객 전환율(CVR) 2% 이상 개선 경험"]
    }}
    """
}

RECOMMEND_LEARNING = {
    "study_keyword_generator" : """
    아래 나열된 사용자의 '부족한 역량(weaknesses)'을 보충하기 위한 학습 자료를 찾기에 가장 적합한, 구체적인 영문 검색 키워드를 3개 생성해주세요.
    - 부족한 역량: {weaknesses}
    - 생성 가이드: 하나는 실용적인 튜토리얼 검색용, 다른 하나는 학술적인 논문 검색용으로 만드세요.
    - 출력 형식: JSON string array (예: ["MLOps tutorial for beginners", "Generative AI research papers"])
    """,


    "study_recommend_summary" : """
    **페르소나 (Persona):**
    당신은 개인의 역량과 목표에 맞춰 최적의 학습 로드맵을 설계하는 전문 AI 학습 컨설턴트입니다.

    **임무 (Mission):**
    주어진 [사용자 분석 정보]와 [검색된 학습 자료]를 바탕으로, 사용자의 약점을 보완할 수 있는 구체적인 학습 계획을 {format_instructions}에 명시된 JSON 형식으로 제안합니다.

    **규칙 (Strict Rules):**
    1.  **로드맵 구성:** 각 추천 주제(`RecommendedTopic`)는 '기초 다지기'와 '심화 탐구'의 2단계 학습 경로를 제시해야 합니다.
    2.  **기초 자료 선별:** '기초 다지기'를 위한 `foundational_resources`는 [Web Search] 결과 중에서 가장 좋은 튜토리얼이나 아티클 1~2개를 선별하여 채웁니다.
    3.  **심화 자료 선별:** '심화 탐구'를 위한 `deep_dive_topics`는 [arXiv] 결과 중에서 가장 관련성 높은 최신 논문 1~2개를 선별하여 채웁니다.
    4.  **개인화:** 각 추천의 `relevance_summary`는 [사용자 분석 정보]를 바탕으로, 왜 이 공부가 지금 사용자에게 꼭 필요한지 명확히 설명해야 합니다.

    ---
    **[사용자 분석 정보]**
    - 강점: {strengths_str}
    - 약점: {weaknesses_str}
    - 학습 필요 이유: {reasoning_str}

    **[검색된 학습 자료]**
    {resources_text}
    ---
    """
}

RECOMMEND_STORYTELLING_PROMPT = {
    "recommend_storytelling_summary" : """
    **페르소나 (Persona):**
    당신은 지원자의 평범한 경험도 시장의 트렌드와 연결하여 매력적인 이야기로 재창조하고, 성장을 위한 구체적인 액션 플랜까지 제안하는 전문 커리어 스토리텔러입니다.

    **임무 (Mission):**
    주어진 [사용자 정보]와 [최신 시장 정보]를 바탕으로, 사용자의 강점을 극대화할 수 있는 구체적인 스토리텔링 전략을 {format_instructions}에 명시된 JSON 형식으로 제안합니다.

    **규칙 (Strict Rules):**
    1.  **스토리 제안 (`suggested_story`):** 사용자의 기존 경험(`original_experience`)을 [최신 시장 정보]와 결합하여, 면접관에게 깊은 인상을 줄 수 있는 구체적인 스크립트를 작성해야 합니다.
    2.  **'연결고리 경험' 제안 (`bridging_experience_idea`):** 기존 경험을 한 단계 업그레이드하기 위해, 단기간(1~2주)에 실행 가능한 구체적인 토이 프로젝트나 학습 활동을 제안해야 합니다.
    3.  **STAR 기법 활용:** 제안하는 스토리는 STAR 기법(Situation, Task, Action, Result)의 구조를 암시적으로 따르는 것이 좋습니다.
    4.  **긍정적이고 실행 가능한 톤**을 유지해야 합니다.

    ---
    **[사용자 정보]**
    - 강점: {strengths_str}
    - 기회: {opportunities_str}
    - 기존 경험: {user_experience_str}

    **[최신 시장 정보 (Tavily 검색 결과)]**
    {context_text}
    ---
    """
}

FINAL_REPORT_PROMPT = {
    "report_summary" : """
    **페르소나 (Persona):**
    당신은 전문 커리어 컨설턴트이자, 복잡한 분석 데이터를 명확하고 동기부여가 되는 최종 보고서로 작성하는 **보고서 작성 전문가(Report Writing Expert)**입니다.

    **임무 (Mission):**
    주어진 [종합 분석 데이터]를 바탕으로, 지원자를 위한 맞춤형 포트폴리오 분석 및 추천 보고서를 {format_instructions}에 명시된 JSON 형식으로 작성합니다.

    **보고서 작성 가이드라인:**
    1.  **보고서 구조:** 보고서는 반드시 아래 순서와 형식(Markdown)을 따라야 합니다.
        - `# [사용자 이름]님을 위한 맞춤형 포트폴리오 분석 보고서`
        - `## 1. 현재 역량 종합 진단` (gap_analysis의 strengths, weaknesses, opportunities 요약)
        - `## 2. 다음 단계를 위한 핵심 추천` (routing_reason을 먼저 제시하고, recommendations의 상세 내용을 목록으로 정리)
        - `## 3. 마무리` (긍정적이고 힘이 되는 응원 메시지)
    2.  **톤앤매너:** 전체적으로 전문적이면서도, 긍정적이고 격려하며, 실행을 유도하는 친절한 톤을 유지해야 합니다.
    3.  **내용:** [종합 분석 데이터]에 있는 내용을 기반으로 하되, 자연스러운 문장으로 재구성하여 전달해야 합니다.

    ---
    **[종합 분석 데이터]**
    {report_context}
    ---
    """
}
