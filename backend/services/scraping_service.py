"""
Product Scraping Service
Service for scraping Amazon product information using scrape_amazon.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from sqlalchemy.orm import Session

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scrape_amazon import scrape_amazon_product, scrape_amazon_reviews
    SCRAPING_AVAILABLE = True
except ImportError as e:
    SCRAPING_AVAILABLE = False
    scrape_amazon_reviews = None
    print(f"Warning: scrape_amazon module not found. Product scraping will not be available.")
    print(f"Import error: {e}")
    print(f"Looking for scrape_amazon.py in: {project_root}")


class ScrapingService:
    """Service for scraping Amazon product information."""
    
    def __init__(self):
        self.available = SCRAPING_AVAILABLE
    
    def scrape_product(self, url: str) -> Optional[Dict]:
        """
        Scrape product information from Amazon URL.
        
        Args:
            url: Amazon product URL
        
        Returns:
            Dictionary with product information or None if scraping fails
        """
        if not self.available:
            raise RuntimeError("Scraping not available. scrape_amazon module not found.")
        
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        if "amazon" not in url.lower():
            raise ValueError("URL must be an Amazon product URL")
        
        scraped_data = scrape_amazon_product(url)
        
        if scraped_data is None:
            return None
        
        product_data = {
            "product_title": scraped_data.get("product_title", ""),
            "product_store_name": scraped_data.get("product_store_name", ""),
            "price": self._parse_price(scraped_data.get("price")),
            "reviews_number": scraped_data.get("num_reviews", 0),
            "average_rating": scraped_data.get("average_rating", 0.0),
            "specs_chars": scraped_data.get("specification_char_count", 0),
            "category": scraped_data.get("category", ""),
            "product_url": url,
            "product_asin": scraped_data.get("product_asin", ""),
            "num_images": scraped_data.get("num_images", 0),
            "description_chars": scraped_data.get("description_char_count", 0),
        }
        
        return product_data
    
    def scrape_reviews(self, url: str, max_reviews: int = 10) -> Optional[List[Dict]]:
        """
        Scrape reviews from an Amazon product URL.
        
        Args:
            url: Amazon product URL
            max_reviews: Maximum number of reviews to scrape
        
        Returns:
            List of review dictionaries or None if scraping fails
        """
        if not self.available:
            raise RuntimeError("Scraping not available. scrape_amazon module not found.")
        
        if scrape_amazon_reviews is None:
            raise RuntimeError("Review scraping function not available.")
        
        if not url or not url.strip():
            raise ValueError("URL cannot be empty")
        
        if "amazon" not in url.lower():
            raise ValueError("URL must be an Amazon product URL")
        
        return scrape_amazon_reviews(url, max_reviews)
    
    def _parse_price(self, price: Optional[str]) -> Optional[float]:
        """Parse price string to float."""
        if price is None:
            return None
        
        try:
            price_cleaned = str(price).replace("$", "").replace(",", "").strip()
            return float(price_cleaned)
        except (ValueError, AttributeError):
            return None
    
    def create_or_update_product(
        self,
        db: Session,
        product_data: Dict,
        product_id: Optional[str] = None
    ) -> Dict:
        """
        Create or update product in database from scraped data.
        Returns product data even if database save fails.
        
        Args:
            db: Database session
            product_data: Scraped product data dictionary
            product_id: Optional product ID (if None, uses ASIN from scraped data)
        
        Returns:
            Dictionary with product information including product_id
        """
        if product_id is None:
            product_id = product_data.get("product_asin")
            if not product_id:
                raise ValueError("product_id or product_asin required")
        
        try:
            from database.models import Product
            
            product = db.query(Product).filter(Product.product_id == product_id).first()
            
            if product:
                product.title = product_data.get("product_title", product.title)
                product.store_name = product_data.get("product_store_name", product.store_name)
                product.price = product_data.get("price", product.price)
                product.reviews_number = product_data.get("reviews_number", product.reviews_number)
                product.average_rating = product_data.get("average_rating", product.average_rating)
                product.specs_chars = product_data.get("specs_chars", product.specs_chars)
                product.category = product_data.get("category", product.category)
                product.product_url = product_data.get("product_url", product.product_url)
            else:
                product = Product(
                    product_id=product_id,
                    title=product_data.get("product_title", "Unknown Product"),
                    store_name=product_data.get("product_store_name", ""),
                    price=product_data.get("price"),
                    reviews_number=product_data.get("reviews_number", 0),
                    average_rating=product_data.get("average_rating", 0.0),
                    specs_chars=product_data.get("specs_chars", 0),
                    category=product_data.get("category", ""),
                    product_url=product_data.get("product_url")
                )
                db.add(product)
            
            db.commit()
            db.refresh(product)
            
            return {
                "product_id": product.product_id,
                "title": product.title,
                "store_name": product.store_name,
                "price": product.price,
                "reviews_number": product.reviews_number,
                "average_rating": product.average_rating,
                "specs_chars": product.specs_chars,
                "category": product.category,
                "product_url": product.product_url
            }
        except Exception as e:
            print(f"⚠️  Could not save product to database: {e}")
            print("⚠️  Returning scraped data without database save")
            
            return {
                "product_id": product_id,
                "title": product_data.get("product_title", "Unknown Product"),
                "store_name": product_data.get("product_store_name", ""),
                "price": product_data.get("price"),
                "reviews_number": product_data.get("reviews_number", 0),
                "average_rating": product_data.get("average_rating", 0.0),
                "specs_chars": product_data.get("specs_chars", 0),
                "category": product_data.get("category", ""),
                "product_url": product_data.get("product_url")
            }

