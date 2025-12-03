"""
Database Initialization Script
Populates the database with sample users and products for testing.
"""

from database.database import init_db, SessionLocal
from database.models import User, Product
import uuid

def create_sample_data():
    """Create sample users and products."""
    db = SessionLocal()
    
    try:
        # Create sample users
        users = [
            User(
                user_id="USER001",
                username="john_doe",
                email="john@example.com",
                review_count=15,
                avg_helpful_votes=3.5
            ),
            User(
                user_id="USER002",
                username="jane_smith",
                email="jane@example.com",
                review_count=42,
                avg_helpful_votes=4.2
            ),
            User(
                user_id="USER003",
                username="tech_reviewer",
                email="tech@example.com",
                review_count=128,
                avg_helpful_votes=4.8
            ),
        ]
        
        # Create sample products
        products = [
            Product(
                product_id="PROD001",
                title="Premium Software Suite",
                store_name="Amazon",
                price=99.99,
                reviews_number=250,
                average_rating=4.5,
                specs_chars=500,
                category="Software"
            ),
            Product(
                product_id="PROD002",
                title="Business Productivity Tool",
                store_name="Microsoft Store",
                price=149.99,
                reviews_number=180,
                average_rating=4.2,
                specs_chars=750,
                category="Business"
            ),
            Product(
                product_id="PROD003",
                title="Creative Design Software",
                store_name="Adobe",
                price=199.99,
                reviews_number=320,
                average_rating=4.7,
                specs_chars=1200,
                category="Design"
            ),
        ]
        
        # Add to database
        for user in users:
            existing = db.query(User).filter(User.user_id == user.user_id).first()
            if not existing:
                db.add(user)
                print(f"✅ Created user: {user.username} ({user.user_id})")
            else:
                print(f"⏭️  User already exists: {user.username}")
        
        for product in products:
            existing = db.query(Product).filter(Product.product_id == product.product_id).first()
            if not existing:
                db.add(product)
                print(f"✅ Created product: {product.title} ({product.product_id})")
            else:
                print(f"⏭️  Product already exists: {product.title}")
        
        db.commit()
        print("\n✅ Database initialized with sample data!")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error initializing database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("\nCreating sample data...")
    create_sample_data()
    print("\n✅ Done!")

