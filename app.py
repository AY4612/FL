import streamlit as st
import pandas as pd
import pyvista as pv
import trimesh
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import base64
import os
import numpy as np
import scipy.spatial as spatial
import tempfile

st.set_page_config(page_title="피부 손상 3D 분석기", layout="wide")

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 800; color: #1a1a2e;
        border-left: 6px solid #e94560; padding-left: 14px; margin-bottom: 4px;
    }
    .sub-title {
        font-size: 0.95rem; color: #555; margin-bottom: 28px; padding-left: 20px;
    }
    .info-box {
        background: #f8f9fc; border: 1px solid #dde3f0;
        border-radius: 10px; padding: 18px 22px; margin-bottom: 20px;
        font-size: 0.88rem; color: #333; line-height: 1.75;
    }
    .info-box b { color: #ff0000; } /* 중요한 글자 빨간색 유지 */
    .section-label {
        font-size: 1.1rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 2px solid #e94560; padding-bottom: 4px; margin: 24px 0 12px;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">속도별 피부 손상 3D 분석</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">바람 속도에 따른 얼굴 피부의 전단응력 · 압력 · TEWL을 3D로 시각화하고, 위험 교집합을 비교합니다.</div>', unsafe_allow_html=True)

# ── 파일 업로드 ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">① 파일 업로드</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    stl_file = st.file_uploader("얼굴 형상 파일",
                               help="CFD 해석에 사용된 원본 얼굴 표면 메시 파일을 올려주세요.")
with col2:
    csv_files = st.file_uploader("속도별 CFD 결과 (CSV) — 3개 모두 업로드",
                                  type=["csv"], accept_multiple_files=True,
                                  help="파일명이 반드시 '5ms', '8ms', '10ms'로 끝나야 자동 정렬됩니다. 예: result_5ms.csv")

# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────
def create_colorbar(vmin, vmax, label_text="", threshold=None):
    fig, ax = plt.subplots(figsize=(6, 0.55))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.45, top=0.95)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.viridis, norm=norm, orientation='horizontal')
    cb.ax.tick_params(labelsize=8)
    cb.set_label(label_text, fontsize=8.5, fontweight='bold', color='#333', labelpad=2)
    
    if threshold is not None and vmax > vmin:
        ax.axvline(x=threshold, color='#ff0000', linewidth=2, linestyle='--')
                
    fig.patch.set_alpha(0.0)
    return fig

def create_colorbar_overlap(label_text=""):
    """중첩 개수에 따른 4단계 커스텀 컬러바 (얇게 수정 및 숫자 라벨 적용)"""
    fig, ax = plt.subplots(figsize=(6, 0.25))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.3, top=0.8)
    
    cmap = mpl.colors.ListedColormap(['#d3d3d3', '#ffcc99', '#ff8c00', '#cc0000'])
    bounds = [0, 1, 2, 3, 4]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal', boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
    
    cb.ax.set_xticklabels(['0', '1', '2', '3'])
    cb.ax.tick_params(labelsize=10, length=0)
    cb.set_label(label_text, fontsize=8.5, fontweight='bold', color='#333', labelpad=4)
    fig.patch.set_alpha(0.0)
    return fig

def normalize(arr, vmin, vmax):
    if vmax == vmin:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

def get_speed_info(file_obj):
    base_name = os.path.splitext(file_obj.name)[0].strip().lower()
    
    if base_name.endswith("5ms"):
        return 1, "5 m/s"
    elif base_name.endswith("8ms"):
        return 2, "8 m/s"
    elif base_name.endswith("10ms"):
        return 3, "10 m/s"
        
    return 99, base_name

def save_glb(vertices, faces, scalars, vmin, vmax, global_threshold):
    clean = np.nan_to_num(scalars, nan=vmin)
    norm_val = np.clip((clean - vmin) / (vmax - vmin) if vmax != vmin else np.zeros_like(clean), 0, 1)
    colors = (cm.viridis(norm_val) * 255).astype(np.uint8)

    hot_idx = clean >= global_threshold
    colors[hot_idx] = [255, 140, 0, 255]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh.fix_normals()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as f:
        glb_path = f.name
    mesh.export(glb_path, file_type='glb')
    return glb_path

def save_glb_overlap(vertices, faces, overlap_count):
    colors = np.full((len(vertices), 4), [211, 211, 211, 255], dtype=np.uint8)
    colors[overlap_count == 1] = [255, 204, 153, 255]
    colors[overlap_count == 2] = [255, 140, 0, 255]
    colors[overlap_count == 3] = [204, 0, 0, 255]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    mesh.fix_normals()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as f:
        glb_path = f.name
    mesh.export(glb_path, file_type='glb')
    return glb_path

def model_viewer_html(b64_data, height=420):
    return f"""
    <script type="module"
        src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js">
    </script>
    <model-viewer
        src="data:model/gltf-binary;base64,{b64_data}"
        camera-controls auto-rotate
        style="width:100%; height:{height}px;
               background: linear-gradient(135deg,#f0f0f0,#dde3f0);
               border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,0.10);">
    </model-viewer>
    """

# ── 세션 상태 초기화 ──────────────────────────────────────────────────────────
if "mapped" not in st.session_state:
    st.session_state.mapped = False

# ── 메인 로직 ─────────────────────────────────────────────────────────────────
csv_count = len(csv_files) if csv_files else 0

if stl_file or csv_count > 0:
    if not stl_file:
        st.warning("STL 파일을 업로드해주세요.")
    elif csv_count < 3:
        st.info(f"CSV 파일 3개(5 · 8 · 10 m/s)를 모두 올려주세요.")
    else:
        st.markdown('<div class="section-label">② 극한 조건 교집합(상위 %) 설정</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        🔴 각 물리량별로 위험하다고 판단할 <b>상위 퍼센트(%) 기준</b>을 설정합니다.<br>
        🔴 슬라이더를 조절하면 아래 3D 모델에 실시간으로 반영되며, 설정한 위험 조건이 <b>겹치는 개수</b>에 따라 색상이 더욱 진하게 통합 표시됩니다.
        </div>
        """, unsafe_allow_html=True)

        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            w_s = st.slider("🩸 전단응력 위험 기준 (상위 %)", 0, 100, 10,
                            help="상위 몇 %의 전단응력을 위험 구역으로 볼지 설정합니다.")
        with w_col2:
            w_p = st.slider("🌬️ 압력 위험 기준 (상위 %)", 0, 100, 10,
                            help="상위 몇 %의 압력을 위험 구역으로 볼지 설정합니다.")
        with w_col3:
            w_t = st.slider("🌡️ TEWL 위험 기준 (상위 %)", 0, 100, 10,
                            help="상위 몇 %의 TEWL 수치를 위험 구역으로 볼지 설정합니다.")

        st.markdown('<div class="section-label">③ 3D 분석 실행</div>', unsafe_allow_html=True)
        if st.button("🚀 3D 변환 및 매핑 시작", type="primary"):
            st.session_state.mapped = True

        if st.session_state.mapped:
            with st.spinner("전체 속도 데이터를 통합 분석하고 3D 모델을 렌더링 중입니다..."):
                try:
                    stl_file.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp:
                        tmp.write(stl_file.read())
                        stl_path = tmp.name

                    surface = pv.read(stl_path)
                    surface = surface.triangulate()
                    vertices = surface.points
                    faces = surface.faces.reshape(-1, 4)[:, 1:]

                    csv_files_sorted = sorted(csv_files, key=lambda x: get_speed_info(x)[0])
                    tab_names = [get_speed_info(f)[1] for f in csv_files_sorted]

                    data_frames = []
                    all_shears, all_pressures, all_tewls = [], [], []

                    for csv_file in csv_files_sorted:
                        csv_file.seek(0)
                        df = pd.read_csv(csv_file)
                        df.columns = df.columns.str.strip()
                        data_frames.append(df)
                        all_shears.append(df['wall-shear'].values)
                        all_pressures.append(df['pressure'].values)
                        
                        # 수정 요청하신 정확한 수식 반영: TEWL = 6.7466 * h
                        h_values = df['heat-transfer-coef'].values
                        tewl_values = 6.7466 * h_values
                        all_tewls.append(tewl_values)

                    all_shears_arr   = np.concatenate(all_shears)
                    all_press_arr    = np.concatenate(all_pressures)
                    all_tewls_arr     = np.concatenate(all_tewls)

                    g_min_shear = float(np.nanpercentile(all_shears_arr, 2))
                    g_max_shear = float(np.nanpercentile(all_shears_arr, 98))
                    g_min_press = float(np.nanpercentile(all_press_arr, 5))
                    g_max_press = float(np.nanpercentile(all_press_arr, 98))
                    g_min_tewl   = float(np.nanpercentile(all_tewls_arr, 2))
                    g_max_tewl   = float(np.nanpercentile(all_tewls_arr, 98))

                    gt_shear = float(np.nanpercentile(all_shears_arr, 100 - w_s))
                    gt_press = float(np.nanpercentile(all_press_arr,  100 - w_p))
                    gt_tewl   = float(np.nanpercentile(all_tewls_arr,   100 - w_t))

                    pre_mapped = []

                    for df_idx, df in enumerate(data_frames):
                        csv_pts = df[['x-coordinate', 'y-coordinate', 'z-coordinate']].values
                        tree = spatial.cKDTree(csv_pts)
                        _, idx = tree.query(vertices)

                        ms = df['wall-shear'].values[idx]
                        mp = df['pressure'].values[idx]
                        
                        # 현재 탭 파일에 해당되는 변환된 TEWL 값 매핑
                        mt = all_tewls[df_idx][idx]

                        mask_s = ms >= gt_shear
                        mask_p = mp >= gt_press
                        mask_t = mt >= gt_tewl
                        
                        overlap_count = mask_s.astype(int) + mask_p.astype(int) + mask_t.astype(int)

                        pre_mapped.append((ms, mp, mt, overlap_count))

                    st.markdown("---")
                    st.markdown("""
                    <div class="info-box">
                    🔴 개별 탭에서는 전체 데이터 색상 분포(보라~노랑)가 표시되며, <b>오렌지색 구역</b>은 설정한 상위 극한 수치를 넘어선 영역입니다.<br>
                    🔴 하단 통합 탭에서는 위험 조건이 <b>겹치는 개수</b>에 따라 색상의 농도가 진해집니다.
                    </div>
                    """, unsafe_allow_html=True)

                    tabs = st.tabs(tab_names)

                    for i, (tab, df) in enumerate(zip(tabs, data_frames)):
                        ms, mp, mt, overlap_count = pre_mapped[i]

                        with tab:
                            col_s, col_p, col_t = st.columns(3)

                            for col, scalars, label, emoji, vmin, vmax, gt, unit, top_pct in [
                                (col_s, ms, "전단응력", "🩸", g_min_shear, g_max_shear, gt_shear, "Shear Stress (Pa)", w_s),
                                (col_p, mp, "압력",     "🌬️", g_min_press, g_max_press, gt_press, "Pressure (Pa)", w_p),
                                (col_t, mt, "TEWL",     "🌡️", g_min_tewl,   g_max_tewl,   gt_tewl,   "TEWL", w_t),
                            ]:
                                with col:
                                    st.markdown(f"**{emoji} {label}**")
                                    fig = create_colorbar(vmin, vmax, unit, threshold=gt if top_pct < 100 else None)
                                    st.pyplot(fig)
                                    plt.close(fig)

                                    over_pct = float(np.mean(scalars >= gt) * 100)
                                    
                                    # 가독성을 높이기 위해 소수점 3자리 표기로 깔끔하게 유지합니다.
                                    st.caption(
                                        f"🔴 표시 범위: {vmin:.3f} ~ {vmax:.3f}  \n"
                                        f"🔴 위험 임계값 (상위 {top_pct}%): **{gt:.3f}** \n"
                                        f"🔴 이 속도에서 위험 구역 면적: **{over_pct:.1f}%**"
                                    )

                                    glb_path = save_glb(vertices, faces, scalars, vmin, vmax, gt)
                                    with open(glb_path, "rb") as f:
                                        b64 = base64.b64encode(f.read()).decode()
                                    os.remove(glb_path)

                                    st.components.v1.html(model_viewer_html(b64, 400), height=420)
                                    
                                    # 모바일 등 세로 배치 시 각 항목 사이의 시각적 간격 띄우기
                                    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

                            st.markdown("---")
                            st.markdown("## 통합 피부 손상 위험도")
                            st.markdown(f"""
                            <div class="info-box">
                            🔴 위에서 설정한 세 가지 극한 조건(<b>전단응력 상위 {w_s}% / 압력 상위 {w_p}% / TEWL 상위 {w_t}%</b>)이 겹치는 영역입니다.<br>
                            🔴 조건이 1개, 2개, 3개 겹칠 때마다 <b>회색 ➔ 연주황 ➔ 오렌지 ➔ 다크 레드</b> 순으로 색상이 깊어집니다.
                            </div>
                            """, unsafe_allow_html=True)
                            
                            fig_c = create_colorbar_overlap("Overlapping Risk Factors")
                            st.pyplot(fig_c)
                            plt.close(fig_c)
                            
                            pct_1 = float(np.mean(overlap_count == 1) * 100)
                            pct_2 = float(np.mean(overlap_count == 2) * 100)
                            pct_3 = float(np.mean(overlap_count == 3) * 100)
                            
                            st.caption(
                                f"🔴 면적 비율  \n"
                                f"1개 겹침: **{pct_1:.1f}%** | 2개 겹침: **{pct_2:.1f}%** | 3개 겹침(최악): **{pct_3:.1f}%**"
                            )

                            glb_c = save_glb_overlap(vertices, faces, overlap_count)
                            with open(glb_c, "rb") as f:
                                b64_c = base64.b64encode(f.read()).decode()
                            os.remove(glb_c)

                            st.components.v1.html(model_viewer_html(b64_c, 600), height=620)

                    st.success("✅ 렌더링 완료! 탭을 이동하며 속도별 위험도를 비교해보세요.")

                except Exception as e:
                    st.error(f"❌ 오류가 발생했습니다: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    if 'stl_path' in locals() and os.path.exists(stl_path):
                        os.remove(stl_path)
