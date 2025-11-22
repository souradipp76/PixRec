import os
import gzip
import json
import requests
from collections import defaultdict

# ---------------------------
# Download the Datasets
# ---------------------------
datasets = [
    {
        "name": "Subscription_Boxes",
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Subscription_Boxes.jsonl.gz",
        "meta_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Subscription_Boxes.jsonl.gz"
    },
    {
        "name": "Magazine_Subscriptions",
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Magazine_Subscriptions.jsonl.gz",
        "meta_url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Magazine_Subscriptions.jsonl.gz"
    }
]

for dataset in datasets:
    # URL for the Amazon 5-core dataset
    review_dataset_url = dataset['url']
    local_review_gz_file = f"data/{dataset['name']}.jsonl.gz"

    if not os.path.exists(local_review_gz_file):
        print("Downloading the Amazon dataset...")
        response = requests.get(review_dataset_url, stream=True)
        with open(local_review_gz_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed.")
    else:
        print("Dataset file already exists.")

    meta_dataset_url = dataset['meta_url']
    local_meta_gz_file = f"data/meta_{dataset['name']}.jsonl.gz"

    if not os.path.exists(local_meta_gz_file):
        print("Downloading the Amazon dataset meta...")
        response = requests.get(meta_dataset_url, stream=True)
        with open(local_meta_gz_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download completed.")
    else:
        print("Dataset Meta file already exists.")

    # Initialize storage for user reviews and unique items
    user_reviews = defaultdict(list)
    unique_items = set()

    items = {}
    with gzip.open(local_meta_gz_file, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            title = item.get("title", "") or "Unknown"
            category = item.get("main_category", "") or "Unknown"
            brand = item.get("details", {}).get("Brand", "") or "Unknown"
            price = item.get("price", "") or "Unknown"

            image_dict = list(filter(lambda x: x['variant'] == 'MAIN', item.get("images")))
            image_url = image_dict[0].get("large") if len(image_dict) == 1 else None
            items[item['parent_asin']] = {
                "title": title,
                "main_category": category,
                "details": {"Brand": brand},
                "price": price,
                "image_url": image_url
            }

    img_dir = f"data/images/{dataset['name']}"
    os.makedirs(img_dir, exist_ok=True)

    count = 0
    print("Processing dataset...")
    with gzip.open(local_review_gz_file, "rt", encoding="utf-8") as f:
        for line in f:
            review = json.loads(line.strip())
            # Ensure the review has the required fields from the new data structure
            if "user_id" not in review or "timestamp" not in review or "title" not in review:
                continue
            user = review["user_id"]
            timestamp = review["timestamp"]
            # Use the review's title as the item title; fallback to "Unknown" if empty.

            item = items.get(review['parent_asin'], {})
            title = item.get("title", "")or "Unknown"
            category = item.get("main_category", "") or "Unknown"
            brand = item.get("details", {}).get("Brand", "") or "Unknown"
            price = item.get("price", "") or "Unknown"
            image_url = item.get("image_url")

            if not image_url:
                continue
            
            img_name = image_url.split("/")[-1]
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                img_data = requests.get(image_url).content
                with open(img_path, 'wb') as f:
                    f.write(img_data)
            # Construct a flattened item text.
            item_text = f"Title: {title}. Category: {category}. Brand: {brand}. Price: {price}. Image: "
            user_reviews[user].append({"timestamp": timestamp, "item_text": item_text, "item_image_path": img_path})
            unique_items.add(item_text)

    # Filter users to keep only those with at least 5 reviews (5-core)
    filtered_users = {user: reviews for user, reviews in user_reviews.items() if len(reviews) >= 2}
    print(f"Total users with >=2 reviews: {len(filtered_users)}")

    output_jsonl = f"data/{dataset['name']}.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for user, reviews in filtered_users.items():
            # Sort reviews by timestamp (ascending)
            reviews_sorted = sorted(reviews, key=lambda x: x["timestamp"])
            user_history = [r["item_text"] for r in reviews_sorted[:-1]]
            user_history_images = [r["item_image_path"] for r in reviews_sorted[:-1]]
            target_item = reviews_sorted[-1]["item_text"]
            target_item_image = reviews_sorted[-1]["item_image_path"]
            record = {"user_history": user_history, "user_history_images": user_history_images, "target_item": target_item, "target_item_image": target_item_image}
            fout.write(json.dumps(record) + "\n")
    print(f"Preprocessed data saved to {output_jsonl}")

    # Write out a corpus file for BM25 retrieval (one unique item per line)
    corpus_file = f"data/amazon_{dataset['name']}_item_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as fcorpus:
        for item in unique_items:
            fcorpus.write(item + "\n")
    print(f"Corpus file saved to {corpus_file}")