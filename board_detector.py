import cv2
import numpy as np
import os

class GridSquareDetector:
    def __init__(self, template_path, threshold=0.95):  # Increased threshold to 0.8 for better precision
        """
        Initialize the GridSquareDetector with a template image.
        
        Args:
            template_path (str): Path to the grid square template image
            threshold (float): Matching threshold (0-1), higher means more strict matching
        """
        print(f"Loading template from: {os.path.abspath(template_path)}")
        self.template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if self.template is None:
            raise ValueError(f"Could not load template image from {os.path.abspath(template_path)}")
            
        print(f"Template size: {self.template.shape[1]}x{self.template.shape[0]} pixels")
        self.threshold = threshold
        self.template_h, self.template_w = self.template.shape[:2]
        self.matched_locations = set()  # Store matched locations to avoid duplicates
        self.match_distance = max(self.template_h, self.template_w) // 2  # Minimum distance between matches
        print(f"Initialized detector with threshold: {self.threshold}, match distance: {self.match_distance}")
        
    def reset(self):
        """Reset the matched locations for a new frame"""
        self.matched_locations.clear()
        
    def find_grid_squares(self, image):
        """
        Find all grid squares in the given image that exactly match the template.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            int: Number of grid squares found
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create a copy of the image for debug visualization
        debug_img = image.copy()
        
        # Convert to grayscale for more robust matching
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching with stricter parameters
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # Set a very high threshold to only get very confident matches
        threshold = 0.95
        loc = np.where(result >= threshold)  # Very strict threshold
        
        # Sort matches by confidence (lower difference is better for SQDIFF)
        matches = []
        for pt in zip(*loc[::-1]):
            x, y = pt[0], pt[1]
            confidence = result[y, x]  # Lower is better for SQDIFF
            # Only consider matches that are well within the image bounds
            # and have a very high confidence score
            if (x > 0 and y > 0 and 
                x + self.template_w < image.shape[1] and 
                y + self.template_h < image.shape[0] and
                confidence > threshold):
                matches.append((x, y, confidence))
                
                # Print debug info for each match
                print(f"Match at ({x}, {y}) with confidence: {confidence:.4f}")
        
        # Sort by confidence (descending for CCOEFF)
        matches.sort(key=lambda m: m[2], reverse=True)
        
        # Track already matched regions
        matched_regions = []
        final_matches = []
        
        for x, y, _ in matches:
            # Define the match region
            match_region = (x, y, x + self.template_w, y + self.template_h)
            
            # Check for overlap with any previously matched region
            overlap = False
            for prev_region in matched_regions:
                # Check if regions overlap
                if not (match_region[2] <= prev_region[0] or  # left
                        match_region[0] >= prev_region[2] or  # right
                        match_region[3] <= prev_region[1] or  # top
                        match_region[1] >= prev_region[3]):   # bottom
                    overlap = True
                    break
            
            if not overlap:
                center_x = int(x + self.template_w/2)
                center_y = int(y + self.template_h/2)
                final_matches.append((center_x, center_y))
                matched_regions.append(match_region)
        
        # Store the matched locations
        self.matched_locations = set(final_matches)
        
        # Draw rectangles around detected squares for debugging
        for x, y in final_matches:
            top_left = (int(x - self.template_w/2), int(y - self.template_h/2))
            bottom_right = (int(x + self.template_w/2), int(y + self.template_h/2))
            cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 2)
            
        # Save the debug image with rectangles
        cv2.imwrite('debug_detection.png', debug_img)
        print(f"Found {len(final_matches)} non-overlapping grid squares")
        print(f"Debug image saved as 'debug_detection.png'")
        return len(final_matches)