from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ml_pipeline import FEATURES, NUMERIC

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data'/'processed'
REPORT = ROOT/'reports'


@st.cache_data
def data():
    return pd.read_csv(DATA/'canonical_all.csv', low_memory=False)


@st.cache_resource
def artifact():
    return joblib.load(ROOT/'modelling'/'model.joblib')


def predict(model, frame):
    if model['baseline']:
        return model['model'].predict(np.zeros((len(frame),1)))
    return model['model'].predict(frame[FEATURES])


st.set_page_config(page_title='Bất động sản Việt Nam', layout='wide')
st.title('Bất động sản Việt Nam')
if not (DATA/'canonical_all.csv').exists() or not (ROOT/'modelling'/'model.joblib').exists():
    st.error('Chưa có dữ liệu hoặc mô hình đã huấn luyện. Vui lòng chạy train.py trước.'); st.stop()
d = data(); model = artifact()
st.caption('Giá chào bán · Đơn vị: tỷ VND · Dữ liệu lịch sử từ tin đăng')
cities = sorted(d.city_province.dropna().unique())
selected_city = st.sidebar.multiselect('Tỉnh / thành phố', cities)
selected_type = st.sidebar.multiselect('Loại nhà', sorted(d.property_type.dropna().unique()))
month_labels = {8: 'Tháng 8/2026', 9: 'Tháng 9/2026', 10: 'Tháng 10/2026', 11: 'Tháng 11/2026'}
selected_months = st.sidebar.multiselect('Tháng đăng tin', list(month_labels.keys()), format_func=lambda m: month_labels.get(m, str(m)))
filtered = d.copy()
if selected_city:
    filtered = filtered[filtered.city_province.isin(selected_city)]
if selected_type:
    filtered = filtered[filtered.property_type.isin(selected_type)]
if selected_months:
    filtered = filtered[filtered.posting_month.isin(selected_months)]
a,b,c = st.columns(3)
a.metric('Tin đăng',f'{len(filtered):,}')
b.metric('Giá trung vị (tỷ VND)',f'{filtered.price.median():.2f}' if filtered.price.notna().any() else '—')
c.metric('Diện tích trung vị (m²)',f'{filtered.area.median():.1f}' if filtered.area.notna().any() else '—')
tab_data, tab_eda, tab_model, tab_predict = st.tabs(['Dữ liệu','Phân tích','Đánh giá','Dự đoán'])
with tab_data:
    cols_show = ['record_id','source','posted_date','city_province','district','property_type','area','price','price_status','eligible_for_model']
    st.dataframe(filtered[[c for c in cols_show if c in filtered.columns]], hide_index=True)
    st.download_button('Tải dữ liệu lọc', filtered.to_csv(index=False).encode('utf-8-sig'),'listings_filtered.csv','text/csv')
    st.dataframe(pd.read_csv(DATA/'missingness_by_source.csv'), hide_index=True)
with tab_eda:
    if filtered.empty:
        st.info('Không có tin phù hợp bộ lọc.')
    else:
        st.subheader('Phân bố Giá & Loại hình BĐS')
        st.plotly_chart(px.histogram(filtered, x='price', nbins=60, labels={'price':'Giá chào bán (tỷ VND)'}), width='stretch')
        counts=filtered.groupby('property_type').size().rename('Số tin').reset_index()
        st.plotly_chart(px.bar(counts,x='property_type',y='Số tin',labels={'property_type':'Loại nhà'}),width='stretch')

        st.subheader('Xu hướng Thời gian (Aug - Nov 2026)')
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            month_counts = filtered.groupby('posting_month').size().rename('Số tin').reset_index()
            month_counts['Tháng'] = month_counts['posting_month'].map(month_labels)
            st.plotly_chart(px.bar(month_counts, x='Tháng', y='Số tin', title='Khối lượng tin đăng theo tháng', color='Tháng'), width='stretch')
        with col_t2:
            price_trend = filtered.groupby('posting_month')['price'].median().rename('Giá trung vị (tỷ)').reset_index()
            price_trend['Tháng'] = price_trend['posting_month'].map(month_labels)
            st.plotly_chart(px.line(price_trend, x='Tháng', y='Giá trung vị (tỷ)', title='Biến động giá trung vị theo tháng', markers=True), width='stretch')

        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_vn = {'Monday':'Thứ 2', 'Tuesday':'Thứ 3', 'Wednesday':'Thứ 4', 'Thursday':'Thứ 5', 'Friday':'Thứ 6', 'Saturday':'Thứ 7', 'Sunday':'Chủ Nhật'}
        if 'day_of_week' in filtered.columns and filtered['day_of_week'].notna().any():
            dow_df = filtered.groupby('day_of_week').size().reindex(dow_order).dropna().rename('Số tin').reset_index()
            dow_df['Thứ'] = dow_df['day_of_week'].map(dow_vn)
            st.plotly_chart(px.bar(dow_df, x='Thứ', y='Số tin', title='Mật độ đăng tin theo ngày trong tuần', color='Thứ'), width='stretch')

        coords=filtered[['latitude','longitude']].dropna()
        if not coords.empty:
            st.caption('Tọa độ từ nguồn tin, chưa xác minh vị trí thực tế. Muaban chưa bổ sung tọa độ.')
            st.map(coords.rename(columns={'latitude':'lat','longitude':'lon'}))
with tab_model:
    st.subheader(model['name'])
    st.caption('Chọn bằng MAE cross-validation theo nhóm; chỉ số test tính trên giá gốc. Không phải tỷ lệ dự đoán đúng.')
    st.dataframe(pd.read_csv(REPORT/'metrics.csv'), hide_index=True)
    st.image(str(REPORT/'test_diagnostics.png'))
    st.dataframe(pd.read_csv(REPORT/'segment_errors.csv'), hide_index=True)
    st.warning('Giá chào bán không phải giá giao dịch. Nguồn gốc Muaban và tính độc lập của mọi tài sản chưa được xác minh hoàn toàn.')
with tab_predict:
    city_value=st.selectbox('Thành phố',cities)
    district_values=sorted(d.loc[d.city_province.eq(city_value),'district'].dropna().unique())
    district_value=st.selectbox('Quận / huyện',district_values or ['Chưa rõ'])
    with st.form('prediction'):
        ptype=st.selectbox('Loại bất động sản',sorted(d.property_type.dropna().unique()))
        area=st.number_input('Diện tích (m²)',min_value=1.,max_value=100000.,value=80.)
        width=st.number_input('Mặt tiền (m)',min_value=0.,max_value=1000.,value=5.)
        width_unknown=st.checkbox('Chưa rõ mặt tiền',value=True)
        rooms={}
        for column,label in [('n_bedrooms','Phòng ngủ'),('n_bathrooms','Phòng tắm'),('n_floors','Số tầng')]:
            val=st.number_input(label,min_value=0,max_value=200,value=2,step=1)
            unknown=st.checkbox('Chưa rõ '+label.lower(),value=True)
            rooms[column]=np.nan if unknown else val
        legal=st.selectbox('Pháp lý',sorted(d.legal.unique()))
        interior=st.selectbox('Nội thất',sorted(d.interior.unique()))
        submit=st.form_submit_button('Ước lượng giá')
    if submit:
        row={'area':area,'front_width':np.nan if width_unknown else width,**rooms,
             'city_province':city_value,'district':district_value,'property_type':ptype,'legal':legal,'interior':interior}
        value=float(predict(model,pd.DataFrame([row]))[0])
        if np.isfinite(value) and value>0:
            st.success(f'Giá chào bán ước lượng: {value:,.2f} tỷ VND')
        else:
            st.error('Mô hình không trả giá hợp lệ cho thông tin này.')
    if st.button('Dự đoán các tin đang lọc', disabled=filtered.empty):
        eligible=filtered[filtered.area.gt(0)].copy()
        if eligible.empty:
            st.info('Không có tin với diện tích hợp lệ.')
        else:
            eligible['predicted_price_billion']=predict(model,eligible)
            st.session_state['batch']=eligible[['record_id','price','predicted_price_billion']]
    if 'batch' in st.session_state:
        st.dataframe(st.session_state['batch'],hide_index=True)
        st.download_button('Tải dự đoán',st.session_state['batch'].to_csv(index=False).encode('utf-8-sig'),'predictions.csv','text/csv')
