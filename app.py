import streamlit as st
import datetime
from PIL import Image
from src.utils.recommendation_engine import RecommendationEngine
from src.models.models import FashionRecommender
from src.utils.model_manager import ModelManager
from src.utils.visualization import draw_detections
from src.utils.body_analysis import BodyAnalyzer
from src.utils.scoring_system import ScoringSystem
from config import MBTI_STYLES

# 전역 변수로 추천 엔진 초기화
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = RecommendationEngine()
if 'fashion_recommender' not in st.session_state:
    st.session_state.fashion_recommender = FashionRecommender()
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'body_analyzer' not in st.session_state:
    st.session_state.body_analyzer = BodyAnalyzer()
if 'scoring_system' not in st.session_state:
    st.session_state.scoring_system = ScoringSystem()




def main():
    """메인함수"""
    st.title("👗 Fitzy - AI 패션 코디 추천")
    st.markdown("업로드한 의상 이미지로 최적의 코디를 추천받아보세요!")
    
    # 사이드바 - 사용자 설정
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # MBTI 선택
        mbti_type = st.selectbox("MBTI 유형", 
                                ["ENFP", "ISTJ", "ESFP", "INTJ"])
        
        # 성별 선택 (자동 인식 기능)
        gender = render_gender_selector()

        # 진단 모드
        debug_mode = st.toggle("🔍 진단 모드 (YOLO/CLIP 상세 분석)", value=False)
        
        # AI 이미지 생성 설정 (선택적)
        with st.expander("🎨 AI 이미지 생성 설정", expanded=False):
            # 초기화 (한 번만)
            if 'enable_ai_images' not in st.session_state:
                st.session_state.enable_ai_images = True
            if 'num_auto_images' not in st.session_state:
                st.session_state.num_auto_images = 1
            
            # 통합된 토글 (활성화 시 자동 생성 포함)
            enable_ai_images = st.toggle(
                "AI 이미지 생성 활성화 (자동 생성 포함)", 
                key="enable_ai_images"
            )
            
            if enable_ai_images:
                # 생성할 이미지 개수 선택
                num_auto_images = st.slider(
                    "자동 생성할 이미지 개수 (추천 코디 중)",
                    min_value=1,
                    max_value=3,
                    key="num_auto_images",
                    help="추천 코디 3개 중 몇 개의 이미지를 자동 생성할지 선택"
                )

        # 날씨 정보 입력
        st.subheader("🌤️ 날씨 정보")
        temperature = st.slider("온도 (°C)", -10, 40, 20)
        weather = st.selectbox("날씨", ["맑음", "흐림", "비", "눈", "바람"])
        
        # 계절 선택
        season = st.selectbox("계절", ["봄", "여름", "가을", "겨울"])
    
    # 메인 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📸 이미지 분석", "🔍 텍스트 검색", "🌟 트렌드 코디", "⚙️ 모델 관리"])
    
    with tab1:
        # 이미지 업로드 및 분석
        uploaded_file = st.file_uploader("옷 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'], key="image_uploader")
        
        # 이미지가 변경되었는지 확인하기 위한 키
        if uploaded_file:
            # 파일이 변경되었는지 확인
            file_id = uploaded_file.name + str(uploaded_file.size)
            if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
                st.session_state.last_file_id = file_id
                # 이미지 관련 캐시 초기화
                if 'processed_image' in st.session_state:
                    del st.session_state.processed_image
                if 'face_info_cache' in st.session_state:
                    del st.session_state.face_info_cache
                if 'body_info_cache' in st.session_state:
                    del st.session_state.body_info_cache
            st.success("이미지 업로드 완료! 분석 중...")
            # 이미지 로드
            image = Image.open(uploaded_file)
            
            # 이미지 표시
            st.image(image, caption="업로드된 이미지", width='stretch')
            processed_image = image
            
            # 얼굴 및 체형 분석
            st.subheader("👤 얼굴 및 체형 분석")
            with st.spinner("얼굴 및 체형 분석 중..."):
                face_info = st.session_state.body_analyzer.analyze_face(processed_image)
                body_info = st.session_state.body_analyzer.analyze_body(processed_image)
                
                # 성별 자동 인식 (이미지가 변경된 경우에만)
                import hashlib
                current_image_hash = hashlib.md5(processed_image.tobytes()).hexdigest()
                
                # last_image_hash 초기화 확인
                if 'last_image_hash' not in st.session_state:
                    st.session_state.last_image_hash = None
                
                # 이미지 해시 저장 (성별 인식은 result 생성 후 수행)
                if current_image_hash != st.session_state.last_image_hash:
                    st.session_state.last_image_hash = current_image_hash
            
            # 분석 결과 표시
            col_face, col_body = st.columns(2)
            with col_face:
                if face_info.get("detected"):
                    st.success("✅ 얼굴 탐지됨")
                    st.write(f"**얼굴 형태:** {face_info.get('face_shape', '알 수 없음')}")
                    st.write(f"**눈 크기:** {face_info.get('eye_size', '알 수 없음')}")
                    
                    # DeepFace 분석 결과 표시
                    if face_info.get("age"):
                        st.write(f"**추정 나이:** {face_info.get('age')}세")
                    if face_info.get("emotion"):
                        emotion_map = {
                            "happy": "😊 행복",
                            "sad": "😢 슬픔",
                            "angry": "😠 화남",
                            "surprise": "😮 놀람",
                            "fear": "😨 두려움",
                            "disgust": "🤢 혐오",
                            "neutral": "😐 무표정"
                        }
                        emotion = face_info.get("emotion", "")
                        emotion_display = emotion_map.get(emotion, emotion)
                        st.write(f"**감정:** {emotion_display}")
                    if face_info.get("gender_deepface"):
                        st.write(f"**DeepFace 성별 인식:** {face_info.get('gender_deepface')}")
                else:
                    st.warning("⚠️ 얼굴을 찾을 수 없습니다")
                    message = face_info.get("message", "얼굴이 명확하게 보이도록 이미지를 업로드해주세요.")
                    st.info(message)
                    if face_info.get("hint"):
                        st.caption(f"💡 {face_info.get('hint')}")
            
            with col_body:
                if body_info.get("detected"):
                    st.success("✅ 체형 분석됨")
                    st.write(f"**체형:** {body_info.get('body_type', '알 수 없음')}")
                else:
                    st.warning("⚠️ 체형을 분석할 수 없습니다")
                    st.info(body_info.get("message", "전신 사진을 업로드해주세요."))
            
            # 코디 추천 결과 표시 (배경 제거 이미지 사용, 얼굴/체형 정보 포함)
            # 먼저 YOLO/CLIP 분석 실행 (점수 계산을 위해)
            fr = st.session_state.fashion_recommender
            result = fr.recommend_outfit(processed_image, mbti_type, temperature, weather, season)
            
            # 성별 자동 인식 (얼굴 특징 기반 + DeepFace + 의상 기반 + CLIP)
            gender_changed = False
            if current_image_hash != st.session_state.get('last_gender_hash', None):
                # 방법 1: 얼굴 특징 기반 성별 인식 (MediaPipe 얼굴 분석 결과 활용)
                # 이미 analyze_face가 호출되어 face_info에 결과가 있음
                detected_gender = None
                
                # 얼굴 특징 기반 추정 시도
                if face_info and face_info.get("detected"):
                    detected_gender = st.session_state.body_analyzer._estimate_gender_from_features(face_info)
                
                # 방법 2: DeepFace 사용 (설치된 경우)
                if not detected_gender:
                    detected_gender = st.session_state.body_analyzer.detect_gender(processed_image)
                
                # 방법 3: 의상 기반 판단
                if not detected_gender:
                    detected_gender = detect_gender_from_image(
                        processed_image, 
                        fr.analyzer,
                        result
                    )
                
                if detected_gender and detected_gender != "공용":
                    # 기존 성별과 비교하여 변경 여부 확인
                    old_gender = st.session_state.get('auto_gender')
                    st.session_state.auto_gender = detected_gender
                    gender_index_map = {"남성": 0, "여성": 1, "공용": 2}
                    new_gender_index = gender_index_map.get(detected_gender, 0)
                    
                    # 성별이 변경되었거나 처음 인식하는 경우
                    if old_gender != detected_gender or st.session_state.selected_gender != new_gender_index:
                        st.session_state.selected_gender = new_gender_index
                        st.session_state.gender_auto_update_pending = True  # rerun 후 업데이트 플래그
                        gender_changed = True
                
                st.session_state.last_gender_hash = current_image_hash
                
                # 성별이 변경되었으면 즉시 사이드바 반영
                if gender_changed:
                    st.rerun()
            
            # 외모 및 패션 점수 계산 (향상된 시스템 사용)
            appearance_scores = st.session_state.scoring_system.score_appearance(
                face_info, body_info, image=processed_image
            )
            fashion_scores = st.session_state.scoring_system.score_fashion(
                result.get("detected_items", {}).get("items", []),
                result.get("style_analysis", {}),
                weather,
                season,
                temperature,
                image=processed_image  # 이미지 전달 (향상된 분석용)
            )
            
            # 점수 표시 (접힌 상태로)
            with st.expander("📊 외모 및 패션 점수", expanded=False):
                col_score1, col_score2 = st.columns(2)
                with col_score1:
                    st.markdown("### 👤 외모 점수")
                    display_score_metric("전체 외모", appearance_scores['전체 외모'])
                
                with col_score2:
                    st.markdown("### 👗 패션 점수")
                    display_score_metric("전체 패션", fashion_scores['전체 패션'])
            
            # 상세 피드백
            feedback = st.session_state.scoring_system.get_detailed_feedback(appearance_scores, fashion_scores, season)
            if feedback:
                with st.expander("💡 개선 제안"):
                    for fb in feedback:
                        st.write(fb)
            
            # 코디 추천 결과 표시
            display_outfit_recommendations(
                processed_image, mbti_type, temperature, weather, season, 
                gender, debug_mode, face_info, body_info, original_image=image,
                precomputed_result=result, appearance_scores=appearance_scores, fashion_scores=fashion_scores
            )
    
    with tab2:
        # 텍스트 기반 코디 검색
        st.subheader("🔍 텍스트 기반 코디 검색")
        
        # 세션 상태 초기화
        if 'search_query' not in st.session_state:
            st.session_state.search_query = ""
        
        # 빠른 선택 버튼
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎉 파티용 코디"):
                st.session_state.search_query = "파티용 코디"
        with col2:
            if st.button("💼 출근룩"):
                st.session_state.search_query = "출근룩"
        with col3:
            if st.button("💕 데이트룩"):
                st.session_state.search_query = "데이트룩"
        
        search_query = st.text_input(
            "원하는 코디를 검색하세요", 
            value=st.session_state.search_query,
            placeholder="예: 파티용 코디, 출근룩, 데이트룩"
        )
        
        if search_query:
            st.session_state.search_query = search_query
            display_text_search_results(search_query, mbti_type)
    
    with tab3:
        # 트렌드 및 인기 코디
        st.subheader("🔥 이번 시즌 인기 코디")
        display_trend_outfits(season)
    
    with tab4:
        # 모델 관리 페이지
        display_model_manager()