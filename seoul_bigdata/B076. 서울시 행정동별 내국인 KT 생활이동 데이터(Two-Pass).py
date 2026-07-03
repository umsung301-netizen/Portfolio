import pandas as pd
import numpy as np
import glob
import os
import gc

# =====================================================================
# [환경 설정] 
# =====================================================================
KT_DIR = './생활이동' 

print("🚀 [SLCI 2단계] 투-패스(Two-Pass) 기반 동적 고립 탐지 엔진 가동 시작\n")

# -----------------------------------------------------------------
# [도우미 함수] 초경량 인코딩 탐지기
# -----------------------------------------------------------------
def detect_encoding(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.readline()
        return 'utf-8'
    except UnicodeDecodeError:
        return 'cp949'
    except Exception:
        return 'utf-8'

# -----------------------------------------------------------------
# [사전 준비] 매칭표 로드 및 서울 거름망 생성
# -----------------------------------------------------------------
print("📦 [준비] 지역코드 매칭표 로드 및 서울 거름망 생성 중...")
mapping_file = glob.glob(f'{KT_DIR}/**/*매칭표*.csv', recursive=True)
if not mapping_file:
    raise FileNotFoundError("🚨 [오류] '지역코드_매칭표.csv'를 찾을 수 없습니다.")

df_mapping = pd.read_csv(mapping_file[0], encoding=detect_encoding(mapping_file[0]))
df_mapping.columns = df_mapping.columns.str.lower().str.strip()
df_mapping['emd'] = df_mapping['emd'].astype(str).str.strip()

# 서울(11)만 남기기
df_mapping = df_mapping[df_mapping['emd'].str.startswith('11')]
df_mapping = df_mapping.drop_duplicates(subset=['emd']).copy()

search_pattern = os.path.join(KT_DIR, '*', '*', '*_OUTPUT.txt')
kt_files = glob.glob(search_pattern)
print(f"🔍 분석 대상 일별 파일 확보: 총 {len(kt_files)}개\n")

# =====================================================================
# 🏃‍♂️ [Pass 1] 첫 번째 훑기: 행정동별 '가중 평균 동내 이동거리' 산출
# =====================================================================
print("🏃‍♂️ [Pass 1] 데이터 1차 스캔: 동네별 평균 이동거리 계산 중...")

target_ages = ['20', '25', '30', '35']
chunk_size = 1000000 
pass1_results = []

for file in kt_files:
    file_encoding = detect_encoding(file)
    try:
        chunk_iter = pd.read_csv(file, sep='|', encoding=file_encoding, chunksize=chunk_size, dtype=str)
    except pd.errors.EmptyDataError:
        continue
        
    for chunk in chunk_iter:
        chunk.columns = chunk.columns.str.lower().str.strip()
        
        # 필수 컬럼 검사
        if not all(c in chunk.columns for c in ['start_emd', 'arv_emd', 'agegrd_nm', 'mvmn_dstc', 'popl_cnt']):
            continue

        # 문자열 공백 정제
        chunk['agegrd_nm'] = chunk['agegrd_nm'].str.strip()
        chunk['start_emd'] = chunk['start_emd'].str.strip()
        chunk['arv_emd'] = chunk['arv_emd'].str.strip()

        # 2030 필터링
        chunk = chunk[chunk['agegrd_nm'].isin(target_ages)].copy()
        
        # [핵심] '동네 안에서의 이동(intra-dong)'만 필터링
        intra_chunk = chunk[chunk['start_emd'] == chunk['arv_emd']].copy()
        if intra_chunk.empty:
            continue
            
        intra_chunk['mvmn_dstc'] = pd.to_numeric(intra_chunk['mvmn_dstc'], errors='coerce').fillna(0)
        intra_chunk['popl_cnt'] = pd.to_numeric(intra_chunk['popl_cnt'], errors='coerce').fillna(0)
        
        # 가중평균을 위한 (거리 * 인구수) 계산
        intra_chunk['weighted_dstc'] = intra_chunk['mvmn_dstc'] * intra_chunk['popl_cnt']
        
        # 청크 단위 요약
        agg = intra_chunk.groupby('start_emd')[['popl_cnt', 'weighted_dstc']].sum().reset_index()
        pass1_results.append(agg)
        
        # 메모리 방어
        if len(pass1_results) >= 100:
            temp = pd.concat(pass1_results).groupby('start_emd').sum().reset_index()
            pass1_results = [temp]

        del chunk, intra_chunk
        gc.collect()

# Pass 1 최종 계산: 각 행정동별 진짜 평균 거리(avg_dstc) 도출
df_pass1 = pd.concat(pass1_results).groupby('start_emd').sum().reset_index()
df_pass1['avg_dstc'] = np.where(df_pass1['popl_cnt'] > 0, df_pass1['weighted_dstc'] / df_pass1['popl_cnt'], 0)

# 두 번째 패스에서 초고속으로 매핑하기 위해 딕셔너리(사전) 형태로 변환
# 예: {'1101053': 1250.5, '1123064': 2100.0, ...}
dong_avg_dict = dict(zip(df_pass1['start_emd'], df_pass1['avg_dstc']))
print("  -> ✅ [Pass 1 완료] 모든 행정동의 2030 평균 동내 이동거리 맵핑 완료!\n")

# =====================================================================
# 🏃‍♂️ [Pass 2] 두 번째 훑기: 동적 임계치를 적용한 고립 스코어링
# =====================================================================
print("🏃‍♂️ [Pass 2] 데이터 2차 스캔: 평균 거리를 기준으로 고립 인구 채점 중...")

pass2_results = []

for file in kt_files:
    file_encoding = detect_encoding(file)
    try:
        chunk_iter = pd.read_csv(file, sep='|', encoding=file_encoding, chunksize=chunk_size, dtype=str)
    except pd.errors.EmptyDataError:
        continue
        
    for chunk in chunk_iter:
        chunk.columns = chunk.columns.str.lower().str.strip()
        
        chunk['agegrd_nm'] = chunk['agegrd_nm'].str.strip()
        chunk['start_emd'] = chunk['start_emd'].str.strip()
        chunk['arv_emd'] = chunk['arv_emd'].str.strip()

        chunk = chunk[chunk['agegrd_nm'].isin(target_ages)].copy()
        if chunk.empty:
            continue
            
        chunk['mvmn_dstc'] = pd.to_numeric(chunk['mvmn_dstc'], errors='coerce').fillna(0)
        chunk['popl_cnt'] = pd.to_numeric(chunk['popl_cnt'], errors='coerce').fillna(0)
        
        # [핵심] 사전(Dictionary)에서 해당 동네의 평균 거리를 빛의 속도로 가져오기
        # 만약 평균값이 없는 동네라면 0으로 처리 (고립 판정 불가)
        chunk['avg_dstc'] = chunk['start_emd'].map(dong_avg_dict).fillna(0)
        
        # [고립 채점] 동내 이동이면서, '그 동네 평균 거리'보다 적게 움직였는가?
        chunk['is_isolated'] = (chunk['start_emd'] == chunk['arv_emd']) & (chunk['mvmn_dstc'] < chunk['avg_dstc'])
        
        chunk['isolated_pop_cnt'] = np.where(chunk['is_isolated'], chunk['popl_cnt'], 0)
        
        # 총 이동량과 고립 이동량 요약
        agg = chunk.groupby('start_emd')[['popl_cnt', 'isolated_pop_cnt']].sum().reset_index()
        pass2_results.append(agg)
        
        # 메모리 방어
        if len(pass2_results) >= 100:
            temp = pd.concat(pass2_results).groupby('start_emd').sum().reset_index()
            pass2_results = [temp]

        del chunk
        gc.collect()

print("  -> ✅ [Pass 2 완료] 전체 고립 인구 채점 완료!\n")

# =====================================================================
# 🗺️ [최종 스코어링 및 반출] 서울 필터링 및 비율 산출
# =====================================================================
print("🗺️ [최종] 2030 행정동별 동적 고립 위험도 결합 및 반출 준비...")

df_total = pd.concat(pass2_results).groupby('start_emd').sum().reset_index()

# 매칭표와 Inner Join (여기서 코드가 '11'로 시작하지 않는 경기도/인천 등 타 지역은 전부 증발)
df_final = pd.merge(df_total, df_mapping[['emd', 'emd_name']], left_on='start_emd', right_on='emd', how='inner')

# 0 나누기 방어 및 최종 비율 계산
df_final['isolation_rate_pct'] = np.where(
    df_final['popl_cnt'] > 0, 
    (df_final['isolated_pop_cnt'] / df_final['popl_cnt']) * 100, 
    0
)

# 보기 좋게 정렬 및 컬럼 정리
df_final = df_final.sort_values(by='isolation_rate_pct', ascending=False).reset_index(drop=True)
df_final = df_final[['emd', 'emd_name', 'popl_cnt', 'isolated_pop_cnt', 'isolation_rate_pct']]
df_final.rename(columns={'popl_cnt': '2030_총_이동량', 'isolated_pop_cnt': '2030_고립_이동량', 'isolation_rate_pct': '고립_비율(%)'}, inplace=True)

print("🎉 모든 분석 완료! 아래는 동적 임계치를 적용한 상위 5개 위험 행정동입니다.\n")
print(df_final.head(5))

# CSV 파일 반출 (한글 깨짐 방지)
export_filename = '2030_동적임계치_행정동별_고립비율.csv'
df_final.to_csv(export_filename, index=False, encoding='utf-8-sig')
print(f"\n💾 전체 서울시 행정동 결과가 성공적으로 저장되었습니다! (파일명: {export_filename})")
