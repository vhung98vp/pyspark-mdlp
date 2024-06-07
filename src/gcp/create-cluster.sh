gcloud dataproc clusters create spark-cluster \
    --enable-component-gateway \
    --region us-central1 \
    --subnet default \
    --no-address \
    --master-machine-type e2-standard-2 \
    --master-boot-disk-type pd-ssd \
    --master-boot-disk-size 50 \
    --num-workers 2 \
    --worker-machine-type e2-standard-2 \
    --worker-boot-disk-type pd-ssd \
    --worker-boot-disk-size 50 \
    --image-version 2.2-ubuntu22 \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --project civil-empire-309810 \
    --service-account 506435147682-compute@developer.gserviceaccount.com


gcloud dataproc clusters create spark-single \
    --enable-component-gateway \
    --region us-central1 \
    --subnet default \
    --no-address \
    --single-node \
    --master-machine-type e2-standard-2 \
    --master-boot-disk-type pd-ssd \
    --master-boot-disk-size 50 \
    --image-version 2.2-ubuntu22 \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --project civil-empire-309810 \
    --service-account 506435147682-compute@developer.gserviceaccount.com