"""
Periodic Retraining Scheduler
Automatically retrains the model with user feedback on a periodic interval.
"""

import schedule
import time
import threading
import os
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from database.database import SessionLocal
from services.online_learning import OnlineLearningService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """Scheduler for periodic model retraining."""
    
    def __init__(self, models_dir: str = "models", retrain_interval_hours: int = 24):
        """
        Initialize retraining scheduler.
        
        Args:
            models_dir: Directory containing model artifacts
            retrain_interval_hours: Hours between retraining attempts (default: 24)
        """
        self.models_dir = Path(models_dir)
        self.retrain_interval_hours = retrain_interval_hours
        self.online_learning = OnlineLearningService(models_dir=str(self.models_dir))
        self.is_running = False
        self.thread = None
    
    def run_retraining(self):
        """Execute model retraining."""
        logger.info(f"Starting scheduled retraining at {datetime.now()}")
        
        db: Session = SessionLocal()
        try:
            dataset_path = os.getenv("DATASET_PATH", "../Software_Cleaned_norm.csv")
            result = self.online_learning.retrain_model(
                db=db,
                original_data_path=dataset_path
            )
            
            if result['status'] == 'success':
                logger.info(f"Retraining completed successfully!")
                logger.info(f"  New version: {result['new_version']}")
                logger.info(f"  Metrics: {result['metrics']}")
                logger.info(f"  Feedback samples: {result['feedback_samples']}")
                
            elif result['status'] == 'insufficient_data':
                logger.info(f"Retraining skipped: {result['message']}")
            else:
                logger.error(f"Retraining failed: {result.get('message', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error during retraining: {e}", exc_info=True)
        finally:
            db.close()
    
    def start(self):
        """Start the scheduler in a background thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info(f"Starting retraining scheduler (interval: {self.retrain_interval_hours} hours)")
        
        schedule.every(self.retrain_interval_hours).hours.do(self.run_retraining)
        
        self.is_running = True
        
        def run_scheduler():
            """Run the scheduler loop."""
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.thread = threading.Thread(target=run_scheduler, daemon=True)
        self.thread.start()
        
        logger.info("Retraining scheduler started successfully")
    
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping retraining scheduler...")
        self.is_running = False
        schedule.clear()
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Retraining scheduler stopped")
    
    def trigger_now(self):
        """Manually trigger retraining immediately."""
        logger.info("Manually triggering retraining...")
        self.run_retraining()


def start_scheduler_in_background(models_dir: str = "models", interval_hours: int = 24):
    """
    Convenience function to start scheduler in background.
    
    Args:
        models_dir: Directory containing model artifacts
        interval_hours: Hours between retraining attempts
    """
    scheduler = RetrainingScheduler(models_dir=models_dir, retrain_interval_hours=interval_hours)
    scheduler.start()
    return scheduler


if __name__ == "__main__":
    print("Starting Retraining Scheduler...")
    print("Press Ctrl+C to stop")
    
    scheduler = RetrainingScheduler(
        models_dir="models",
        retrain_interval_hours=24  # Retrain daily
    )
    
    try:
        scheduler.start()
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        scheduler.stop()
        print("Scheduler stopped.")

