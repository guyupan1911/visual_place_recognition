MAPPING_ROOT_DIR=/storage/xray_data/pacifica-cn-229/20230924/kapture_data
QUERY_ROOT_DIR=/storage/xray_data/pacifica-cn-229/20230918/kapture_data
OUTPUT_FILE=data/pairs_mapping_20.txt
GLOBAL_FEAT_DESC=Resnet101-AP-GeM-LM18
GLOBAL_FEAT_TOPK=20  # number of retrieved images for mapping and localization

kapture_compute_image_pairs.py -v debug \
  --mapping ${MAPPING_ROOT_DIR} \
  --query ${QUERY_ROOT_DIR} \
  --global-features-type ${GLOBAL_FEAT_DESC} \
  --topk ${GLOBAL_FEAT_TOPK} \
  --output ${OUTPUT_FILE}
