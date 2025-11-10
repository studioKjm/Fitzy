"""
채팅봇 유틸리티 - OpenAI API를 사용한 코디 추천 챗봇
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, Any

# 환경변수 로드
load_dotenv()

class FashionChatbot:
    """패션 코디 추천 챗봇"""
    
    def __init__(self):
        """OpenAI 클라이언트 초기화"""
        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("ENDPOINT_URL")
        deployment = os.getenv("DEPLOYMENT_NAME")
        api_version = os.getenv("API_VERSION", "2025-01-01-preview")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 환경변수에 설정되지 않았습니다.")
        
        # Azure OpenAI 또는 일반 OpenAI 설정
        if endpoint and deployment:
            # Azure OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=f"{endpoint}openai/deployments/{deployment}",
                default_query={"api-version": api_version}
            )
            self.model = deployment
        else:
            # 일반 OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4"
        
        # 시스템 프롬프트
        self.system_prompt = """당신은 전문 패션 코디 추천 어시스턴트입니다. 
사용자의 요청에 따라 패션 코디를 추천하고, 스타일링 조언을 제공합니다.

주요 기능:
1. 코디 추천: 사용자의 상황(날씨, 계절, 장소, 스타일 선호도)에 맞는 코디 추천
2. 스타일링 조언: 색상 조합, 아이템 매칭, 트렌드 정보 제공
3. 개인화 추천: MBTI, 성별, 선호 스타일을 고려한 맞춤형 추천

응답 형식:
- 친절하고 전문적인 톤으로 답변
- 구체적인 아이템명과 색상 제시
- 실용적이고 실행 가능한 조언 제공
- 한국어로 답변"""
    
    def get_recommendation_context(self, 
                                   gender: Optional[str] = None,
                                   mbti: Optional[str] = None,
                                   temperature: Optional[float] = None,
                                   weather: Optional[str] = None,
                                   season: Optional[str] = None,
                                   detected_items: Optional[list] = None) -> str:
        """현재 사용자 컨텍스트를 문자열로 변환"""
        context_parts = []
        
        if gender:
            context_parts.append(f"성별: {gender}")
        if mbti:
            context_parts.append(f"MBTI: {mbti}")
        if temperature is not None:
            context_parts.append(f"온도: {temperature}°C")
        if weather:
            context_parts.append(f"날씨: {weather}")
        if season:
            context_parts.append(f"계절: {season}")
        if detected_items:
            context_parts.append(f"탐지된 아이템: {', '.join(detected_items)}")
        
        if context_parts:
            return "현재 사용자 정보: " + ", ".join(context_parts)
        return ""
    
    def chat(self, 
             message: str,
             conversation_history: list = None,
             user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        챗봇과 대화
        
        Args:
            message: 사용자 메시지
            conversation_history: 이전 대화 기록 [{"role": "user/assistant", "content": "..."}]
            user_context: 사용자 컨텍스트 (성별, MBTI, 날씨 등)
        
        Returns:
            챗봇 응답
        """
        try:
            # 메시지 구성
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # 사용자 컨텍스트 추가
            if user_context:
                context_str = self.get_recommendation_context(**user_context)
                if context_str:
                    messages.append({
                        "role": "system",
                        "content": f"{context_str}\n\n이 정보를 참고하여 사용자에게 맞는 코디를 추천해주세요."
                    })
            
            # 대화 기록 추가
            if conversation_history:
                messages.extend(conversation_history)
            
            # 현재 사용자 메시지 추가
            messages.append({"role": "user", "content": message})
            
            # API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.9,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

