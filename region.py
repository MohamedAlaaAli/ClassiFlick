def apply_region_growing(self):
    """
    Perform region growing segmentation.

    Parameters:
        image (numpy.ndarray): Input image.
        seeds (list): List of seed points (x, y).
        threshold (float): Threshold for similarity measure.

    Returns:
        numpy.ndarray: Segmented image.
    """
    # Initialize visited mask and segmented image

    # 'visited' is initialized to keep track of which pixels have been visited (Mask)
    visited = np.zeros_like(self.rg_input_grayscale, dtype=bool)
    # 'segmented' will store the segmented image where each pixel belonging
    # to a region will be marked with the corresponding color
    segmented = np.zeros_like(self.rg_input)

    # Define 3x3 window for mean calculation
    window_size = 3
    half_window = window_size // 2

    # Loop through seed points
    for seed in self.rg_seeds:
        seed_x, seed_y = seed

        # Check if seed coordinates are within image bounds
        if (
                0 <= seed_x < self.rg_input_grayscale.shape[0]
                and 0 <= seed_y < self.rg_input_grayscale.shape[1]
        ):
            # Process the seed point
            region_mean = self.rg_input_grayscale[seed_x, seed_y]

        # Initialize region queue with seed point
        # It holds the candidate pixels
        queue = [(seed_x, seed_y)]

        # Region growing loop
        # - Breadth-First Search (BFS) is used here to ensure
        # that all similar pixels are added to the region
        while queue:
            # Pop pixel from queue
            x, y = queue.pop(0)

            # Check if pixel is within image bounds and not visited
            if (
                    (0 <= x < self.rg_input_grayscale.shape[0])
                    and (0 <= y < self.rg_input_grayscale.shape[1])
                    and not visited[x, y]
            ):
                # Mark pixel as visited
                visited[x, y] = True

                # Check similarity with region mean
                if (
                        abs(self.rg_input_grayscale[x, y] - region_mean)
                        <= self.rg_threshold
                ):
                    # Add pixel to region
                    segmented[x, y] = self.rg_input[x, y]

                    # Add neighbors to queue
                    for i in range(-half_window, half_window + 1):
                        for j in range(-half_window, half_window + 1):
                            if (
                                    0 <= x + i < self.rg_input_grayscale.shape[0]
                                    and 0 <= y + j < self.rg_input_grayscale.shape[1]
                            ):
                                queue.append((x + i, y + j))

    self.plot_rg_output(segmented)
    # self.display_image(segmented, self.ui.sift_output_figure_canvas, "SIFT Output")


def plot_rg_output(self, segmented_image):
    ## =========== Display the segmented image =========== ##
    # Find contours of segmented region
    contours, _ = cv2.findContours(
        cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Draw contours on input image
    output_image = self.rg_input.copy()
    cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

    # Display the output image
    self.display_image(
        output_image,
        self.ui.region_growing_output_figure_canvas,
        "Region Growing Output",
    )
