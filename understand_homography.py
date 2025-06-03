import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw  # Added ImageDraw here

# --- Matplotlib Imports for 3D plotting ---
import matplotlib

matplotlib.use('TkAgg')  # Specify TkAgg backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # For drawing 3D polygons


class HomographyVisualizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Homography Visualizer with 3D Plot")
        master.geometry("1000x900")  # Adjusted window size for 3D plot

        # --- Configuration ---
        self.canvas_width = 280  # Slightly reduced to fit better
        self.canvas_height = 280
        self.grid_step = 20
        self.point_radius = 4
        self.line_width = 2

        # --- Data Storage ---
        self.src_points_entries = []
        self.dst_points_entries = []
        self.homography_matrix_labels = []
        self.H = None  # To store the calculated homography matrix

        # --- Create base grid image ---
        self.base_grid_image_pil = self._create_grid_pil(self.canvas_width, self.canvas_height, self.grid_step)
        self.base_grid_image_cv = self._pil_to_cv2(self.base_grid_image_pil.copy())

        # --- Show Vectors Option ---
        self.show_vectors_var = tk.BooleanVar(value=False)

        # --- UI Setup ---
        self._setup_ui()

        # --- Initial visualization ---
        self._draw_initial_canvases()
        self._update_3d_plot(None, None)  # Initialize 3D plot

    def _pil_to_cv2(self, pil_image):
        """Converts a PIL image to an OpenCV image."""
        open_cv_image = np.array(pil_image)
        if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:  # RGB
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
        elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:  # RGBA
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGRA)
        return open_cv_image

    def _cv2_to_pil(self, cv_image):
        """Converts an OpenCV image to a PIL image."""
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:  # BGR
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:  # BGRA
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
        else:  # Grayscale
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(cv_image_rgb)

    def _create_grid_pil(self, width, height, step):
        """Creates a PIL Image with a grid."""
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)  # ImageDraw is used here
        for x in range(0, width, step):
            draw.line([(x, 0), (x, height)], fill="lightgray")
        for y in range(0, height, step):
            draw.line([(0, y), (width, y)], fill="lightgray")
        return image

    def _setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel: Inputs and Controls ---
        left_panel = ttk.Frame(main_frame, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Source Points
        src_frame = ttk.LabelFrame(left_panel, text="Source Points (x, y)", padding="10")
        src_frame.pack(pady=5, fill=tk.X)
        default_src_pts = [(50, 50), (200, 50), (200, 200), (50, 200)]  # Adjusted defaults for smaller canvas
        for i in range(4):
            row_frame = ttk.Frame(src_frame)
            row_frame.pack()
            ttk.Label(row_frame, text=f"P{i + 1}:").pack(side=tk.LEFT)
            entry_x = ttk.Entry(row_frame, width=5)
            entry_x.insert(0, str(default_src_pts[i][0]))
            entry_x.pack(side=tk.LEFT, padx=2)
            entry_y = ttk.Entry(row_frame, width=5)
            entry_y.insert(0, str(default_src_pts[i][1]))
            entry_y.pack(side=tk.LEFT, padx=2)
            self.src_points_entries.append((entry_x, entry_y))

        # Destination Points
        dst_frame = ttk.LabelFrame(left_panel, text="Destination Points (x', y')", padding="10")
        dst_frame.pack(pady=5, fill=tk.X)
        default_dst_pts = [(70, 70), (230, 60), (240, 180), (60, 190)]  # Adjusted defaults
        for i in range(4):
            row_frame = ttk.Frame(dst_frame)
            row_frame.pack()
            ttk.Label(row_frame, text=f"P'{i + 1}:").pack(side=tk.LEFT)
            entry_x = ttk.Entry(row_frame, width=5)
            entry_x.insert(0, str(default_dst_pts[i][0]))
            entry_x.pack(side=tk.LEFT, padx=2)
            entry_y = ttk.Entry(row_frame, width=5)
            entry_y.insert(0, str(default_dst_pts[i][1]))
            entry_y.pack(side=tk.LEFT, padx=2)
            self.dst_points_entries.append((entry_x, entry_y))

        # Show Vectors Checkbox
        vector_checkbox = ttk.Checkbutton(left_panel, text="Show Vectors (2D Origin)", variable=self.show_vectors_var,
                                          command=self._on_compute_homography)
        vector_checkbox.pack(pady=10)

        # Compute Button
        compute_button = ttk.Button(left_panel, text="Compute Homography", command=self._on_compute_homography)
        compute_button.pack(pady=10)

        # Homography Matrix Display
        matrix_frame = ttk.LabelFrame(left_panel, text="Homography Matrix (H)", padding="10")
        matrix_frame.pack(pady=5, fill=tk.X)
        for r in range(3):
            row_frame = ttk.Frame(matrix_frame)
            row_frame.pack()
            row_labels = []
            for c in range(3):
                lbl = ttk.Label(row_frame, text="0.000", width=8, relief="sunken", anchor="e")
                lbl.pack(side=tk.LEFT, padx=2, pady=2)
                row_labels.append(lbl)
            self.homography_matrix_labels.append(row_labels)

        # --- Right Panel: Visualizations ---
        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Top part of right panel for 2D canvases
        top_right_panel = ttk.Frame(right_panel)
        top_right_panel.pack(fill=tk.X, expand=False)  # Don't expand vertically

        # Source Canvas
        src_canvas_frame = ttk.LabelFrame(top_right_panel, text="Source Plane", padding="5")
        src_canvas_frame.pack(pady=5, padx=5, side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.source_canvas = tk.Canvas(src_canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white",
                                       relief="ridge", borderwidth=2)
        self.source_canvas.pack()

        # Destination Canvas
        dst_canvas_frame = ttk.LabelFrame(top_right_panel, text="Destination Plane (Warped)", padding="5")
        dst_canvas_frame.pack(pady=5, padx=5, side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.destination_canvas = tk.Canvas(dst_canvas_frame, width=self.canvas_width, height=self.canvas_height,
                                            bg="white", relief="ridge", borderwidth=2)
        self.destination_canvas.pack()

        # Bottom part of right panel for 3D plot
        bottom_right_panel = ttk.LabelFrame(right_panel, text="3D Visualization", padding="5")
        bottom_right_panel.pack(fill=tk.BOTH, expand=True, pady=10)

        self.fig_3d = Figure(figsize=(6, 5), dpi=100)  # Adjust size as needed
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        self.canvas_3d_widget = FigureCanvasTkAgg(self.fig_3d, master=bottom_right_panel)
        self.canvas_3d_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # self.canvas_3d_widget.draw() # Initial draw is handled in _update_3d_plot

    def _get_points_from_entries(self):
        src_pts = []
        dst_pts = []
        try:
            for ex, ey in self.src_points_entries:
                src_pts.append((float(ex.get()), float(ey.get())))
            for ex, ey in self.dst_points_entries:
                dst_pts.append((float(ex.get()), float(ey.get())))
        except ValueError:
            messagebox.showerror("Input Error", "Invalid input. Please enter numeric coordinates.")
            return None, None

        if len(src_pts) != 4 or len(dst_pts) != 4:
            messagebox.showerror("Input Error", "Please enter exactly 4 source and 4 destination points.")
            return None, None

        return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)

    def _update_homography_display(self):
        if self.H is not None:
            for r in range(3):
                for c in range(3):
                    self.homography_matrix_labels[r][c].config(text=f"{self.H[r, c]:.4f}")
        else:
            for r in range(3):
                for c in range(3):
                    self.homography_matrix_labels[r][c].config(text="N/A")

    def _draw_quad_and_points(self, canvas, points_coords, tk_image_bg, color="blue", point_color="red", tag_prefix=""):
        canvas.delete(f"{tag_prefix}_quad")
        canvas.delete(f"{tag_prefix}_points")
        canvas.delete(f"{tag_prefix}_vectors")
        canvas.delete(f"{tag_prefix}_bg")  # Ensure old bg is cleared before new one

        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image_bg, tags=f"{tag_prefix}_bg")
        canvas.image = tk_image_bg

        if points_coords is None or len(points_coords) < 4:
            return

        for i in range(4):
            p1 = points_coords[i]
            p2 = points_coords[(i + 1) % 4]
            canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=self.line_width, tags=f"{tag_prefix}_quad")

        for x, y in points_coords:
            canvas.create_oval(x - self.point_radius, y - self.point_radius,
                               x + self.point_radius, y + self.point_radius,
                               fill=point_color, outline=point_color, tags=f"{tag_prefix}_points")

        if self.show_vectors_var.get():
            origin = (0, 0)
            if tag_prefix == "src":
                for pt in points_coords:
                    canvas.create_line(origin[0], origin[1], pt[0], pt[1], fill="darkgreen", width=1, arrow=tk.LAST,
                                       tags=f"{tag_prefix}_vectors")
            elif tag_prefix == "dst" and self.H is not None:
                origin_homogeneous = np.array([0, 0, 1], dtype=np.float32)
                transformed_origin_homogeneous = self.H @ origin_homogeneous

                transformed_origin_pt = (float('inf'), float('inf'))  # Default if division by zero
                if abs(transformed_origin_homogeneous[2]) > 1e-6:  # Avoid division by zero or very small numbers
                    transformed_origin_pt = (transformed_origin_homogeneous[0] / transformed_origin_homogeneous[2],
                                             transformed_origin_homogeneous[1] / transformed_origin_homogeneous[2])

                src_pts_for_vectors, _ = self._get_points_from_entries()
                if src_pts_for_vectors is not None:
                    for pt_src in src_pts_for_vectors:
                        pt_src_homogeneous = np.array([pt_src[0], pt_src[1], 1], dtype=np.float32)
                        transformed_pt_homogeneous = self.H @ pt_src_homogeneous

                        transformed_pt = (float('inf'), float('inf'))  # Default
                        if abs(transformed_pt_homogeneous[2]) > 1e-6:
                            transformed_pt = (transformed_pt_homogeneous[0] / transformed_pt_homogeneous[2],
                                              transformed_pt_homogeneous[1] / transformed_pt_homogeneous[2])

                        if all(abs(c) < self.canvas_width * 5 for c in transformed_origin_pt) and \
                                all(abs(c) < self.canvas_height * 5 for c in transformed_pt):  # Generous bounds check
                            canvas.create_line(transformed_origin_pt[0], transformed_origin_pt[1],
                                               transformed_pt[0], transformed_pt[1],
                                               fill="purple", width=1, arrow=tk.LAST, tags=f"{tag_prefix}_vectors")

    def _draw_initial_canvases(self):
        src_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil)
        self.source_canvas.create_image(0, 0, anchor=tk.NW, image=src_img_tk, tags="src_bg")
        self.source_canvas.image = src_img_tk

        dst_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil.copy())
        self.destination_canvas.create_image(0, 0, anchor=tk.NW, image=dst_img_tk, tags="dst_bg")
        self.destination_canvas.image = dst_img_tk

    def _update_3d_plot(self, src_pts_2d, dst_pts_2d_user_input):
        """Updates the 3D matplotlib plot."""
        self.ax_3d.clear()

        if src_pts_2d is not None and dst_pts_2d_user_input is not None and \
                len(src_pts_2d) == 4 and len(dst_pts_2d_user_input) == 4:

            # Source points in 3D (on z=0 plane)
            src_pts_3d = np.concatenate([src_pts_2d, np.zeros((4, 1))], axis=1)
            # Destination points in 3D (on z=1 plane - arbitrary choice for visualization)
            dst_pts_3d = np.concatenate([dst_pts_2d_user_input, np.ones((4, 1))], axis=1)

            # Create Poly3DCollection for source quad
            # Vertices must be in order for the polygon: (p0, p1, p2, p3)
            verts_src = [
                list(zip(src_pts_3d[[0, 1, 2, 3], 0], src_pts_3d[[0, 1, 2, 3], 1], src_pts_3d[[0, 1, 2, 3], 2]))]
            poly_src = Poly3DCollection(verts_src, alpha=0.6, facecolors='cyan', linewidths=1.5, edgecolors='blue')
            self.ax_3d.add_collection3d(poly_src)

            # Create Poly3DCollection for destination quad
            verts_dst = [
                list(zip(dst_pts_3d[[0, 1, 2, 3], 0], dst_pts_3d[[0, 1, 2, 3], 1], dst_pts_3d[[0, 1, 2, 3], 2]))]
            poly_dst = Poly3DCollection(verts_dst, alpha=0.6, facecolors='lightgreen', linewidths=1.5,
                                        edgecolors='green')
            self.ax_3d.add_collection3d(poly_dst)

            # Draw lines connecting corresponding points (projection rays)
            for i in range(4):
                self.ax_3d.plot([src_pts_3d[i, 0], dst_pts_3d[i, 0]],
                                [src_pts_3d[i, 1], dst_pts_3d[i, 1]],
                                [src_pts_3d[i, 2], dst_pts_3d[i, 2]], 'r--', linewidth=1)

            # Set plot limits and labels
            all_x = np.concatenate([src_pts_3d[:, 0], dst_pts_3d[:, 0]])
            all_y = np.concatenate([src_pts_3d[:, 1], dst_pts_3d[:, 1]])

            min_x, max_x = np.min(all_x) - self.grid_step, np.max(all_x) + self.grid_step
            min_y, max_y = np.min(all_y) - self.grid_step, np.max(all_y) + self.grid_step

            self.ax_3d.set_xlim([min(0, min_x), max(self.canvas_width, max_x)])
            self.ax_3d.set_ylim([min(0, min_y), max(self.canvas_height, max_y)])
            self.ax_3d.set_zlim([-0.5, 1.5])

            self.ax_3d.set_xlabel('X axis')
            self.ax_3d.set_ylabel('Y axis')
            self.ax_3d.set_zlabel('Z (Conceptual Plane)')
            self.ax_3d.set_title('3D Plane-to-Plane Mapping')
        else:
            self.ax_3d.set_title('3D Visualization (Enter valid points)')
            self.ax_3d.set_xlabel('X')
            self.ax_3d.set_ylabel('Y')
            self.ax_3d.set_zlabel('Z')
            # Set some default view if no points
            self.ax_3d.set_xlim([0, self.canvas_width])
            self.ax_3d.set_ylim([0, self.canvas_height])
            self.ax_3d.set_zlim([-0.5, 1.5])

        self.canvas_3d_widget.draw()

    def _on_compute_homography(self):
        src_pts, dst_pts = self._get_points_from_entries()

        if src_pts is None or dst_pts is None:
            self.H = None
            self._update_homography_display()
            self._draw_initial_canvases()
            # Try to draw source points if available, even if dst are bad or homography fails
            current_src_pts, _ = self._get_points_from_entries()  # Re-fetch, might have corrected one set
            if current_src_pts is not None:
                src_img_tk_temp = ImageTk.PhotoImage(self.base_grid_image_pil)
                self._draw_quad_and_points(self.source_canvas, current_src_pts, src_img_tk_temp, color="blue",
                                           point_color="red", tag_prefix="src")
            # Clear destination canvas specifically if points were bad
            dst_img_tk_base = ImageTk.PhotoImage(self.base_grid_image_pil.copy())
            self.destination_canvas.create_image(0, 0, anchor=tk.NW, image=dst_img_tk_base, tags="dst_bg")
            self.destination_canvas.image = dst_img_tk_base
            self.destination_canvas.delete("dst_quad")
            self.destination_canvas.delete("dst_points")
            self.destination_canvas.delete("dst_vectors")

            self._update_3d_plot(current_src_pts, None)  # Update 3D plot with what we have
            return

        H_matrix, mask = cv2.findHomography(src_pts, dst_pts, 0)

        if H_matrix is None:
            messagebox.showerror("Error",
                                 "Could not compute Homography. Check if points are collinear or too few unique points.")
            self.H = None
        else:
            self.H = H_matrix

        self._update_homography_display()

        # --- Update Source Canvas ---
        src_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil)
        self._draw_quad_and_points(self.source_canvas, src_pts, src_img_tk, color="blue", point_color="red",
                                   tag_prefix="src")

        # --- Update Destination Canvas ---
        if self.H is not None:
            warped_grid_cv = cv2.warpPerspective(self.base_grid_image_cv, self.H,
                                                 (self.canvas_width, self.canvas_height))
            warped_grid_pil = self._cv2_to_pil(warped_grid_cv)
            dst_img_tk = ImageTk.PhotoImage(warped_grid_pil)
            self._draw_quad_and_points(self.destination_canvas, dst_pts, dst_img_tk, color="green",
                                       point_color="orange", tag_prefix="dst")
        else:  # If H is None, just draw dst_pts on a plain grid
            dst_img_tk_plain = ImageTk.PhotoImage(self.base_grid_image_pil.copy())
            self._draw_quad_and_points(self.destination_canvas, dst_pts, dst_img_tk_plain, color="green",
                                       point_color="orange", tag_prefix="dst")

        # --- Update 3D Plot ---
        self._update_3d_plot(src_pts, dst_pts)


if __name__ == "__main__":
    root = tk.Tk()
    app = HomographyVisualizerApp(root)
    root.mainloop()