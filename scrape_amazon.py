import re
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


def _extract_numeric_price(price_text: str) -> Optional[str]:
    if not price_text:
        return None
    
    price_cleaned = re.sub(r"[A-Z]{2,3}\s*", "", price_text, flags=re.IGNORECASE)
    price_cleaned = re.sub(r"[$£€¥₹₽₩₪₫₨₦₡₵₴₸₶₷₺₼₾₿]", "", price_cleaned)
    price_cleaned = re.sub(r"[^\d.,]", "", price_cleaned)
    
    price_cleaned = price_cleaned.replace(",", "")
    
    try:
        float(price_cleaned)
        return price_cleaned
    except ValueError:
        return None


def scrape_amazon_product(url: str) -> Optional[Dict[str, object]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "DNT": "1",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
    except Exception:
        return None

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "lxml")

    title_el = soup.select_one("#productTitle")
    product_title = _clean_text(title_el.get_text()) if title_el else None

    store_el = soup.select_one("#bylineInfo") or soup.select_one("#bylineInfo_feature_div")
    product_store_name = _clean_text(store_el.get_text()) if store_el else None

    breadcrumb_links = soup.select(
        "#wayfinding-breadcrumbs_feature_div ul li span a, "
        "#wayfinding-breadcrumbs_container ul li span a"
    )
    categories_found = [
        _clean_text(link.get_text()).upper() for link in breadcrumb_links if link.get_text().strip()
    ]
    
    category: Optional[str] = None
    subcategory: Optional[str] = None
    
    if categories_found:
        skip_items = ["HOME", "ALL", "DEPARTMENTS", "AMAZON"]
        filtered_categories = [cat for cat in categories_found if cat not in skip_items]
        
        if len(filtered_categories) >= 1:
            category = filtered_categories[0]
        if len(filtered_categories) >= 2:
            subcategory = filtered_categories[1]
        elif len(filtered_categories) == 1:
            category = filtered_categories[0]
    
    predefined = [
        "AMAZON FASHION",
        "APPSTORE FOR ANDROID",
        "BOOKS",
        "COMPUTERS",
        "GIFT CARDS",
        "HOME AUDIO & THEATER",
        "SOFTWARE",
        "TOYS & GAMES",
    ]
    category_flags: Dict[str, int] = {}
    for cat in predefined:
        flag = int(any(cat in c or c in cat for c in categories_found))
        key = f"cat_{cat.replace(' ', '_').replace('&', 'and')}"
        category_flags[key] = flag

    product_has_video = False
    num_videos: Optional[int] = None
    
    video_count_el = soup.find(id="videoCount")
    if not video_count_el:
        video_count_el = soup.find(class_=re.compile(r"video-count", re.I))
    
    if video_count_el:
        video_count_text = _clean_text(video_count_el.get_text())
        video_match = re.search(r"(\d+)\s*(?:VIDEO|VIDEOS)", video_count_text, re.IGNORECASE)
        if video_match:
            num_videos = int(video_match.group(1))
            product_has_video = num_videos > 0
        else:
            video_match = re.search(r"(\d+)", video_count_text)
            if video_match:
                num_videos = int(video_match.group(1))
                product_has_video = num_videos > 0
    
    if not product_has_video:
        video_count_by_class = soup.find(class_="video-count")
        if video_count_by_class:
            video_count_text = _clean_text(video_count_by_class.get_text())
            video_match = re.search(r"(\d+)\s*(?:VIDEO|VIDEOS)", video_count_text, re.IGNORECASE)
            if video_match:
                num_videos = int(video_match.group(1))
                product_has_video = num_videos > 0
    
    if not product_has_video:
        product_video_selectors = [
            "#dv-product-video",
            "#dv-product-video-player",
            "#product-video",
            ".product-video",
            "#dv-dp-video",
            "#dv-dp-video-player",
            "div[data-video-id]",
            "#aiv-content-video",
            "#iv-video-container",
            "#iv-video",
        ]
        
        for selector in product_video_selectors:
            video_el = soup.select_one(selector)
            if video_el:
                product_has_video = True
                break
    
    if not product_has_video:
        video_sections = [
            "#ivLargeVideo",
            "#iv-video-container",
            "#imageBlock_feature_div",
            "#altImages",
            "#landingImage",
        ]
        for section_sel in video_sections:
            section = soup.select_one(section_sel)
            if section:
                video_tags = section.find_all("video")
                if video_tags:
                    product_has_video = True
                    if num_videos is None:
                        num_videos = len(video_tags)
                    break
                video_data_attrs = section.find_all(attrs={"data-video-id": True})
                if video_data_attrs:
                    product_has_video = True
                    if num_videos is None:
                        num_videos = len(video_data_attrs)
                    break

    spec_sections = []
    for selector in [
        "#productOverview_feature_div",
        "#productOverview_feature_div table",
        "#prodDetails",
        "#productDetails_techSpec_section_1",
        "#productDetails_detailBullets_sections1",
    ]:
        section = soup.select_one(selector)
        if section:
            spec_sections.append(section)
            break
    spec_text = " "
    if spec_sections:
        spec_text = _clean_text(spec_sections[0].get_text(separator=" "))
    specification_char_count = len(spec_text)

    desc_section = soup.select_one("#productDescription") or soup.select_one("#feature-bullets")
    description_char_count = 0
    if desc_section:
        desc_text = _clean_text(desc_section.get_text(separator=" "))
        description_char_count = len(desc_text)

    asin: Optional[str] = None
    detail_bullets_text = " ".join(
        _clean_text(li.get_text(separator=" ")) for li in soup.select("#detailBullets_feature_div li")
    )
    m = re.search(r"ASIN\s*[:]?\s*([A-Z0-9]{10})", detail_bullets_text)
    if m:
        asin = m.group(1)
    if not asin:
        m = re.search(r"/dp/([A-Z0-9]{10})", url)
        if m:
            asin = m.group(1)
        if not asin:
            m = re.search(r"/gp/product/([A-Z0-9]{10})", url)
            if m:
                asin = m.group(1)

    num_images = 0
    alt_images_container = soup.select_one("#altImages")
    if alt_images_container:
        thumbnail_items = alt_images_container.select("li.imageThumbnail, li.item")
        visible_thumbnails = [
            item for item in thumbnail_items
            if "aok-hidden" not in item.get("class", [])
            and "template" not in " ".join(item.get("class", [])).lower()
            and "videoCountTemplate" not in " ".join(item.get("class", []))
            and "360IngressTemplate" not in " ".join(item.get("class", []))
        ]
        num_images = len(visible_thumbnails)
        
        container_html = str(alt_images_container)
        
        more_match = re.search(r'class="[^"]*textMoreImages[^"]*"[^>]*>(\d+)\+</span>', container_html)
        
        if not more_match:
            more_match = re.search(r'textMoreImages[^>]*>(\d+)\+', container_html)
        
        if not more_match:
            more_match = re.search(r'textMoreImages.*?(\d+)\+', container_html, re.DOTALL)
        
        if more_match:
            additional_images = int(more_match.group(1))
            num_images += additional_images
    
    if product_has_video and num_images > 0:
        num_images -= 1
    
    if num_images == 0:
        image_block_images = soup.select("#imageBlock_feature_div img")
        if image_block_images:
            num_images = len(image_block_images)

    num_reviews: Optional[int] = None
    review_selectors = [
        "#acrCustomerReviewText",
        "#acrCustomerReviewLink",
        "a[data-hook='see-all-reviews-link-foot']",
        "#reviewsMedley .a-size-base",
    ]
    
    for selector in review_selectors:
        reviews_el = soup.select_one(selector)
        if reviews_el:
            reviews_text = _clean_text(reviews_el.get_text())
            digits = re.search(r"([\d,]+)", reviews_text)
            if digits:
                num_reviews = int(digits.group(1).replace(",", ""))
                break

    average_rating: Optional[float] = None
    rating_el = soup.select_one("#acrPopover")
    if rating_el and rating_el.has_attr("title"):
        rating_text = rating_el["title"]
        match = re.search(r"([0-9.]+)", rating_text)
        if match:
            try:
                average_rating = float(match.group(1))
            except ValueError:
                average_rating = None

    price: Optional[str] = None
    price_selectors_offscreen = [
        "span.a-price.aok-align-center.reinventPricePriceToPayMargin.priceToPay span.a-offscreen",
        "span.a-price[data-a-size='xl'] span.a-offscreen",
        "span.a-price span.a-offscreen",
        "#corePrice_feature_div span.a-offscreen",
    ]
    
    for sel in price_selectors_offscreen:
        price_el = soup.select_one(sel)
        if price_el:
            price_text = _clean_text(price_el.get_text())
            if price_text and price_text.strip() and price_text != "$0":
                price = _extract_numeric_price(price_text)
                if price:
                    break
    
    if not price:
        price_containers = [
            "span.a-price.aok-align-center.reinventPricePriceToPayMargin.priceToPay",
            "span.a-price[data-a-size='xl']",
            ".a-price.aok-align-center",
        ]
        
        for container_sel in price_containers:
            price_container = soup.select_one(container_sel)
            if price_container:
                symbol_el = price_container.select_one(".a-price-symbol")
                whole_el = price_container.select_one(".a-price-whole")
                fraction_el = price_container.select_one(".a-price-fraction")
                
                if whole_el or fraction_el:
                    whole = _clean_text(whole_el.get_text()) if whole_el else ""
                    fraction = _clean_text(fraction_el.get_text()) if fraction_el else ""
                    
                    if whole:
                        whole = whole.replace(",", "")
                        if fraction:
                            price = f"{whole}.{fraction}"
                        else:
                            price = whole
                        break
    
    if not price:
        old_price_selectors = [
            "#priceblock_ourprice",
            "#priceblock_dealprice",
            ".a-price-whole",
            "#price",
        ]
        for sel in old_price_selectors:
            price_el = soup.select_one(sel)
            if price_el:
                price_text = _clean_text(price_el.get_text())
                if price_text and price_text.strip():
                    price = _extract_numeric_price(price_text)
                    if price:
                        break

    result: Dict[str, object] = {
        "product_title": product_title,
        "product_has_video": product_has_video,
        "num_videos": num_videos,
        "product_store_name": product_store_name,
        "category": category,
        "subcategory": subcategory,
        **category_flags,
        "specification_char_count": specification_char_count,
        "description_char_count": description_char_count,
        "product_asin": asin,
        "num_images": num_images,
        "num_reviews": num_reviews,
        "average_rating": average_rating,
        "price": price,
    }
    return result


def scrape_amazon_reviews(url: str, max_reviews: int = 10) -> Optional[List[Dict[str, object]]]:
    """
    Scrape reviews from an Amazon product page.
    
    Args:
        url: Amazon product URL
        max_reviews: Maximum number of reviews to scrape (default: 10)
    
    Returns:
        List of review dictionaries or None if scraping fails
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "DNT": "1",
    }
    
    try:
        # Extract ASIN from URL
        asin_match = re.search(r"/dp/([A-Z0-9]{10})", url)
        if not asin_match:
            asin_match = re.search(r"/gp/product/([A-Z0-9]{10})", url)
        if not asin_match:
            return None
        
        asin = asin_match.group(1)
        
        # Construct reviews URL
        # Amazon.eg uses different review URL format
        if "amazon.eg" in url.lower():
            reviews_url = f"https://www.amazon.eg/-/en/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
        else:
            reviews_url = f"https://www.amazon.com/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
        
        response = requests.get(reviews_url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, "lxml")
        reviews = []
        
        # Find review elements - Amazon uses various selectors
        review_elements = soup.select("[data-hook='review']")
        
        if not review_elements:
            # Try alternative selectors
            review_elements = soup.select(".a-section.review")
        
        if not review_elements:
            review_elements = soup.select("#cm_cr-review_list .review")
        
        for review_el in review_elements[:max_reviews]:
            try:
                # Extract review title
                title_el = review_el.select_one("[data-hook='review-title']")
                if not title_el:
                    title_el = review_el.select_one(".a-text-bold span")
                review_title = _clean_text(title_el.get_text()) if title_el else ""
                
                # Remove "Verified Purchase" or rating from title if present
                if review_title:
                    review_title = re.sub(r"^\d+\.\d+ out of \d+ stars\s*", "", review_title)
                    review_title = re.sub(r"\s*Verified Purchase.*$", "", review_title)
                    review_title = review_title.strip()
                
                # Extract review text
                text_el = review_el.select_one("[data-hook='review-body'] span")
                if not text_el:
                    text_el = review_el.select_one(".review-text-content span")
                if not text_el:
                    text_el = review_el.select_one(".a-size-base.review-text")
                review_text = _clean_text(text_el.get_text()) if text_el else ""
                
                # Extract rating
                rating = None
                rating_el = review_el.select_one("[data-hook='review-star-rating']")
                if not rating_el:
                    rating_el = review_el.select_one(".a-icon-alt")
                if rating_el:
                    rating_text = rating_el.get_text() if hasattr(rating_el, 'get_text') else str(rating_el)
                    rating_match = re.search(r"(\d+)\.?\d*\s*out of", rating_text)
                    if rating_match:
                        rating = int(rating_match.group(1))
                
                # Extract verified purchase
                verified = False
                verified_el = review_el.select_one("[data-hook='avp-badge']")
                if not verified_el:
                    verified_text = review_el.get_text()
                    verified = "Verified Purchase" in verified_text or "تم التحقق من الشراء" in verified_text
                else:
                    verified = True
                
                # Extract review date
                date_el = review_el.select_one("[data-hook='review-date']")
                if not date_el:
                    date_el = review_el.select_one(".review-date")
                review_date = _clean_text(date_el.get_text()) if date_el else ""
                
                # Extract reviewer name
                reviewer_el = review_el.select_one("[data-hook='review-author']")
                if not reviewer_el:
                    reviewer_el = review_el.select_one(".a-profile-name")
                reviewer_name = _clean_text(reviewer_el.get_text()) if reviewer_el else ""
                
                if review_text:  # Only add if we have review text
                    reviews.append({
                        "review_title": review_title,
                        "review_text": review_text,
                        "rating": rating or 0,
                        "verified_purchase": verified,
                        "review_date": review_date,
                        "reviewer_name": reviewer_name,
                        "product_asin": asin
                    })
            except Exception as e:
                # Skip reviews that fail to parse
                continue
        
        return reviews if reviews else None
        
    except Exception as e:
        print(f"Error scraping reviews: {e}")
        return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scrape_amazon.py <amazon_product_url>")
    else:
        url_arg = sys.argv[1]
        data = scrape_amazon_product(url_arg)
        if data is None:
            print("Failed to scrape the provided URL.")
        else:
            for k, v in data.items():
                print(f"{k}: {v}")
