MAPPING_ROOT_DIR=/storage/xray_data/yuemeite/mapping/kapture_data
QUERY_ROOT_DIR=/storage/xray_data/yuemeite/query/kapture_data
OUTPUT_FILE=data/ApGem_yuemeite_pairs_mapping_20.txt
GLOBAL_FEAT_DESC=Resnet101-AP-GeM-LM18
GLOBAL_FEAT_TOPK=20  # number of retrieved images for mapping and localization

kapture_compute_image_pairs.py -v debug \
  --mapping ${MAPPING_ROOT_DIR} \
  --query ${QUERY_ROOT_DIR} \
  --global-features-type ${GLOBAL_FEAT_DESC} \
  --topk ${GLOBAL_FEAT_TOPK} \
  --output ${OUTPUT_FILE}
