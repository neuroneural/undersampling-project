mkdir -p /data/users2/jwardell1/undersampling-project/OULU/pkl-files/demo-exp/08-14/rwp/not_mixed

cd /data/users2/jwardell1/undersampling-project/OULU/pkl-files/demo-exp/08-14/rwp/

# Copy files with timestamp 1755197266 or older
for f in *_2025-08-14-175519*.pkl; do
    # Extract timestamp from filename
    ts=$(echo "$f" | grep -oP '\d{10}(?=_usrate)')
    if [ "$ts" -le 1755197266 ]; then
        cp "$f" not_mixed/
    fi
done

