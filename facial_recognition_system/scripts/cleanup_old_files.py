#!/usr/bin/env python3
"""
Clean up old files (logs, captured frames, reports)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta


def get_file_age_days(file_path):
    """Get file age in days"""
    mtime = os.path.getmtime(file_path)
    age_seconds = time.time() - mtime
    return age_seconds / (24 * 3600)


def cleanup_directory(directory, pattern, max_age_days, dry_run=False):
    """Clean up old files in a directory"""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 0
    
    deleted_count = 0
    saved_space = 0
    
    for file_path in Path(directory).glob(pattern):
        if file_path.is_file():
            age_days = get_file_age_days(file_path)
            
            if age_days > max_age_days:
                size = file_path.stat().st_size
                
                if dry_run:
                    print(f"Would delete: {file_path} ({age_days:.1f} days old, {size/1024:.1f} KB)")
                else:
                    print(f"Deleting: {file_path} ({age_days:.1f} days old, {size/1024:.1f} KB)")
                    file_path.unlink()
                
                deleted_count += 1
                saved_space += size
    
    return deleted_count, saved_space


def main():
    parser = argparse.ArgumentParser(description='Clean up old files')
    parser.add_argument('--logs-dir', default='data/logs',
                       help='Logs directory (default: data/logs)')
    parser.add_argument('--frames-dir', default='captured_frames',
                       help='Captured frames directory (default: captured_frames)')
    parser.add_argument('--reports-dir', default='reports',
                       help='Reports directory (default: reports)')
    parser.add_argument('--log-age', type=int, default=30,
                       help='Max age for log files in days (default: 30)')
    parser.add_argument('--frame-age', type=int, default=7,
                       help='Max age for captured frames in days (default: 7)')
    parser.add_argument('--report-age', type=int, default=90,
                       help='Max age for reports in days (default: 90)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--all', action='store_true',
                       help='Clean up all directories')
    
    args = parser.parse_args()
    
    total_deleted = 0
    total_space_saved = 0
    
    print(f"Starting cleanup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be actually deleted")
    print("-" * 60)
    
    # Clean up logs
    if args.all or args.logs_dir:
        print(f"\nCleaning up logs older than {args.log_age} days...")
        count, space = cleanup_directory(
            args.logs_dir, 
            "*.log*", 
            args.log_age,
            args.dry_run
        )
        total_deleted += count
        total_space_saved += space
        print(f"Deleted {count} log files, saved {space/1024/1024:.2f} MB")
    
    # Clean up captured frames
    if args.all or args.frames_dir:
        print(f"\nCleaning up captured frames older than {args.frame_age} days...")
        count, space = cleanup_directory(
            args.frames_dir, 
            "*.jpg", 
            args.frame_age,
            args.dry_run
        )
        total_deleted += count
        total_space_saved += space
        print(f"Deleted {count} frame files, saved {space/1024/1024:.2f} MB")
    
    # Clean up reports
    if args.all or args.reports_dir:
        print(f"\nCleaning up reports older than {args.report_age} days...")
        count, space = cleanup_directory(
            args.reports_dir, 
            "*.json", 
            args.report_age,
            args.dry_run
        )
        total_deleted += count
        total_space_saved += space
        print(f"Deleted {count} report files, saved {space/1024/1024:.2f} MB")
    
    # Clean up temporary files
    temp_patterns = ['*.tmp', '*.temp', '*.bak']
    for pattern in temp_patterns:
        count, space = cleanup_directory('.', pattern, 1, args.dry_run)
        total_deleted += count
        total_space_saved += space
    
    print("\n" + "=" * 60)
    print(f"Cleanup completed!")
    print(f"Total files deleted: {total_deleted}")
    print(f"Total space saved: {total_space_saved/1024/1024:.2f} MB")
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'dry_run': args.dry_run,
        'files_deleted': total_deleted,
        'space_saved_mb': total_space_saved / 1024 / 1024,
        'settings': {
            'log_age_days': args.log_age,
            'frame_age_days': args.frame_age,
            'report_age_days': args.report_age
        }
    }
    
    # Save report
    report_path = f"reports/cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('reports', exist_ok=True)
    
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Cleanup report saved to {report_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())