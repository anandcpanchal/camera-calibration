import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class HomographyVisualizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Homography Visualizer")
        master.geometry("1000x950")

        self.canvas_width = 280
        self.canvas_height = 280
        self.grid_step = 20
        self.point_radius = 4
        self.line_width = 2

        # --- Point count ---
        self.num_points_var = tk.IntVar(value=4)
        self.max_points = 10  # Maximum number of points

        self.src_points_entries = []
        self.dst_points_entries = []
        # Store frames for point entries for easier clearing
        self.src_point_entry_frames_container = None
        self.dst_point_entry_frames_container = None

        self.homography_matrix_entries = []
        self.H = np.eye(3)

        self.base_grid_image_pil = self._create_grid_pil(self.canvas_width, self.canvas_height, self.grid_step)
        self.base_grid_image_cv = self._pil_to_cv2(self.base_grid_image_pil.copy())
        self.show_vectors_var = tk.BooleanVar(value=False)

        # Default points (up to max_points)
        self.default_src_pts_data = [
            (50, 50), (200, 50), (200, 200), (50, 200),
            (25, 125), (225, 125), (125, 25), (125, 225),
            (75, 75), (175, 175)
        ]
        self.default_dst_pts_data = [
            (70, 70), (230, 60), (240, 180), (60, 190),
            (45, 135), (215, 145), (135, 45), (145, 215),
            (85, 85), (185, 185)
        ]

        self._setup_ui()
        self._update_homography_display()  # Initialize H matrix entries with identity
        self._draw_initial_canvases()  # Draw base grids on 2D canvases

        # Compute H from default points and trigger the first full visual draw
        self._on_compute_homography_from_points()

    def _pil_to_cv2(self, pil_image):
        open_cv_image = np.array(pil_image)
        if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:  # RGB
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
        elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:  # RGBA
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGRA)
        return open_cv_image

    def _cv2_to_pil(self, cv_image):
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:  # BGR
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:  # BGRA
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
        else:  # Grayscale
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(cv_image_rgb)

    def _create_grid_pil(self, width, height, step):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        for x in range(0, width, step):
            draw.line([(x, 0), (x, height)], fill="lightgray")
        for y in range(0, height, step):
            draw.line([(0, y), (width, y)], fill="lightgray")
        return image

    def _setup_ui(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # --- Point Count Selector ---
        point_count_frame = ttk.Frame(left_panel)
        point_count_frame.pack(pady=5, fill=tk.X)
        ttk.Label(point_count_frame, text="Number of Points:").pack(side=tk.LEFT, padx=(0, 5))
        self.point_count_spinbox = ttk.Spinbox(
            point_count_frame,
            from_=4,
            to=self.max_points,
            textvariable=self.num_points_var,
            width=3,
            command=self._on_num_points_changed,  # Called when spinbox is changed by user
            state='readonly'
        )
        self.point_count_spinbox.pack(side=tk.LEFT)
        # Bind to variable change for programmatic changes or direct entry (if not readonly)
        self.num_points_var.trace_add("write", self._on_num_points_changed_by_var)

        # --- Source Points ---
        src_frame = ttk.LabelFrame(left_panel, text="Source Points (x, y)", padding="10")
        src_frame.pack(pady=5, fill=tk.X)
        self.src_point_entry_frames_container = ttk.Frame(src_frame)  # Container for dynamic rows
        self.src_point_entry_frames_container.pack(fill=tk.X)

        # --- Destination Points ---
        dst_frame = ttk.LabelFrame(left_panel, text="Destination Points (x', y')", padding="10")
        dst_frame.pack(pady=5, fill=tk.X)
        self.dst_point_entry_frames_container = ttk.Frame(dst_frame)  # Container for dynamic rows
        self.dst_point_entry_frames_container.pack(fill=tk.X)

        # Initial creation of point entry rows (does not trigger full redraw itself)
        self._create_point_entry_rows(self.num_points_var.get())

        vector_checkbox = ttk.Checkbutton(left_panel, text="Show Vectors (2D Origin)", variable=self.show_vectors_var,
                                          command=self._trigger_redraw_visuals)
        vector_checkbox.pack(pady=5)

        compute_button = ttk.Button(left_panel, text="Compute H from Points",
                                    command=self._on_compute_homography_from_points)
        compute_button.pack(pady=(10, 0))

        matrix_frame = ttk.LabelFrame(left_panel, text="Homography Matrix (H) - Editable", padding="10")
        matrix_frame.pack(pady=5, fill=tk.X)
        for r in range(3):
            row_frame = ttk.Frame(matrix_frame)
            row_frame.pack()
            row_entries = []
            for c in range(3):
                entry_h = ttk.Entry(row_frame, width=8, justify='right')
                entry_h.insert(0, f"{self.H[r, c]:.4f}")  # Initial H is identity
                entry_h.pack(side=tk.LEFT, padx=2, pady=2)
                row_entries.append(entry_h)
            self.homography_matrix_entries.append(row_entries)

        apply_h_button = ttk.Button(left_panel, text="Apply Manual H & Update Dst Pts",
                                    command=self._on_apply_manual_homography)
        apply_h_button.pack(pady=(0, 10))

        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        top_right_panel = ttk.Frame(right_panel)
        top_right_panel.pack(fill=tk.X, expand=False)  # Does not expand vertically

        src_canvas_frame = ttk.LabelFrame(top_right_panel, text="Source Plane", padding="5")
        src_canvas_frame.pack(pady=5, padx=5, side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.source_canvas = tk.Canvas(src_canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white",
                                       relief="ridge", borderwidth=2)
        self.source_canvas.pack()

        dst_canvas_frame = ttk.LabelFrame(top_right_panel, text="Destination Plane (Warped)", padding="5")
        dst_canvas_frame.pack(pady=5, padx=5, side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.destination_canvas = tk.Canvas(dst_canvas_frame, width=self.canvas_width, height=self.canvas_height,
                                            bg="white", relief="ridge", borderwidth=2)
        self.destination_canvas.pack()

        bottom_right_panel = ttk.LabelFrame(right_panel, text="3D Visualization", padding="5")
        bottom_right_panel.pack(fill=tk.BOTH, expand=True, pady=10)  # Expands vertically

        self.fig_3d = Figure(figsize=(6, 5), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d_widget = FigureCanvasTkAgg(self.fig_3d, master=bottom_right_panel)
        self.canvas_3d_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _create_point_entry_rows(self, num_points_to_create):
        """Dynamically creates or updates rows for point entries. Does NOT trigger full redraw."""
        # Clear existing source point entries
        for widget in self.src_point_entry_frames_container.winfo_children():
            widget.destroy()
        self.src_points_entries.clear()

        # Clear existing destination point entries
        for widget in self.dst_point_entry_frames_container.winfo_children():
            widget.destroy()
        self.dst_points_entries.clear()

        # Create new source point entries
        for i in range(num_points_to_create):
            row_frame = ttk.Frame(self.src_point_entry_frames_container)
            row_frame.pack()
            ttk.Label(row_frame, text=f"P{i + 1}:").pack(side=tk.LEFT)
            entry_x = ttk.Entry(row_frame, width=5)
            entry_x.insert(0, str(self.default_src_pts_data[i % len(self.default_src_pts_data)][0]))
            entry_x.pack(side=tk.LEFT, padx=2)
            entry_y = ttk.Entry(row_frame, width=5)
            entry_y.insert(0, str(self.default_src_pts_data[i % len(self.default_src_pts_data)][1]))
            entry_y.pack(side=tk.LEFT, padx=2)
            self.src_points_entries.append((entry_x, entry_y))

        # Create new destination point entries
        for i in range(num_points_to_create):
            row_frame = ttk.Frame(self.dst_point_entry_frames_container)
            row_frame.pack()
            ttk.Label(row_frame, text=f"P'{i + 1}:").pack(side=tk.LEFT)
            entry_x = ttk.Entry(row_frame, width=5)
            entry_x.insert(0, str(self.default_dst_pts_data[i % len(self.default_dst_pts_data)][0]))
            entry_x.pack(side=tk.LEFT, padx=2)
            entry_y = ttk.Entry(row_frame, width=5)
            entry_y.insert(0, str(self.default_dst_pts_data[i % len(self.default_dst_pts_data)][1]))
            entry_y.pack(side=tk.LEFT, padx=2)
            self.dst_points_entries.append((entry_x, entry_y))

        # DO NOT call _trigger_redraw_visuals() here. Let the calling function decide.

    def _on_num_points_changed_by_var(self, *args):
        """Handles num_points change if triggered by variable write."""
        try:
            current_ui_num_entries = len(self.src_points_entries)
            new_num_points_from_var = self.num_points_var.get()

            if new_num_points_from_var < 4:
                self.num_points_var.set(4)  # This will re-trigger this callback
                return
            if new_num_points_from_var > self.max_points:
                self.num_points_var.set(self.max_points)  # This will re-trigger this callback
                return

            # Only recreate rows and redraw if the number of entries actually needs to change
            if current_ui_num_entries != new_num_points_from_var:
                self._create_point_entry_rows(new_num_points_from_var)
                self._trigger_redraw_visuals()  # Redraw after rows are recreated
        except tk.TclError:
            pass

    def _on_num_points_changed(self):
        """Handles num_points change from Spinbox user interaction."""
        try:
            new_num_points = self.num_points_var.get()
            # Spinbox itself should ensure value is within its from_/to range.
            # No need to re-validate against self.max_points here if spinbox 'to' is self.max_points.

            self._create_point_entry_rows(new_num_points)
            self._trigger_redraw_visuals()  # Redraw after rows are recreated
        except tk.TclError:
            pass

    def _get_points_from_entries(self, point_entries_list):
        pts = []
        num_expected_points = self.num_points_var.get()

        entries_to_read = min(len(point_entries_list), num_expected_points)

        try:
            for i in range(entries_to_read):
                ex, ey = point_entries_list[i]
                pts.append((float(ex.get()), float(ey.get())))
        except ValueError:
            messagebox.showerror("Input Error", "Invalid numeric coordinates in point entries.")
            return None

        if len(pts) != num_expected_points:
            messagebox.showerror("Input Error",
                                 f"Expected {num_expected_points} points, but found {len(pts)} valid entries.")
            return None
        if len(pts) < 4:  # This check might be redundant if num_expected_points is already >= 4
            messagebox.showerror("Input Error", f"At least 4 points are required for homography, got {len(pts)}.")
            return None
        return np.array(pts, dtype=np.float32)

    def _update_point_entries(self, point_entries_ui_list, points_data_array):
        if points_data_array is None:
            return

        num_to_update = min(len(point_entries_ui_list), len(points_data_array))

        for i in range(num_to_update):
            ex, ey = point_entries_ui_list[i]
            ex.delete(0, tk.END)
            ex.insert(0, f"{points_data_array[i, 0]:.2f}")
            ey.delete(0, tk.END)
            ey.insert(0, f"{points_data_array[i, 1]:.2f}")

    def _update_homography_display(self):
        if self.H is not None:
            for r in range(3):
                for c in range(3):
                    entry_widget = self.homography_matrix_entries[r][c]
                    entry_widget.delete(0, tk.END)
                    entry_widget.insert(0, f"{self.H[r, c]:.4f}")
        else:
            for r in range(3):
                for c in range(3):
                    entry_widget = self.homography_matrix_entries[r][c]
                    entry_widget.delete(0, tk.END)
                    entry_widget.insert(0, "N/A")

    def _draw_quad_and_points(self, canvas, points_coords_array, tk_image_bg, color="blue", point_color="red",
                              tag_prefix=""):
        # Clear previous drawings for this specific canvas based on tag_prefix
        canvas.delete(f"{tag_prefix}_poly")
        canvas.delete(f"{tag_prefix}_points")
        canvas.delete(f"{tag_prefix}_vectors")
        canvas.delete(f"{tag_prefix}_bg")  # Clear old background image too

        # Display background image (grid)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image_bg, tags=f"{tag_prefix}_bg")
        canvas.image = tk_image_bg  # Keep a reference to prevent garbage collection

        if points_coords_array is None or len(points_coords_array) < 2:
            return

        num_current_points = len(points_coords_array)

        # Draw polygon (closed loop if 3+ points)
        if num_current_points >= 2:
            flat_points = []
            for pt in points_coords_array:
                flat_points.extend(pt)  # [(x1,y1), (x2,y2)] -> [x1,y1,x2,y2]

            if num_current_points >= 3:  # Draw polygon for 3 or more points
                canvas.create_polygon(flat_points, fill="", outline=color, width=self.line_width,
                                      tags=f"{tag_prefix}_poly")
            elif num_current_points == 2:  # Draw a line if only 2 points
                canvas.create_line(flat_points[0], flat_points[1], flat_points[2], flat_points[3], fill=color,
                                   width=self.line_width, tags=f"{tag_prefix}_poly")

        # Draw individual points
        for x, y in points_coords_array:
            canvas.create_oval(x - self.point_radius, y - self.point_radius,
                               x + self.point_radius, y + self.point_radius,
                               fill=point_color, outline=point_color, tags=f"{tag_prefix}_points")

        # Optionally draw vectors from origin
        if self.show_vectors_var.get():
            origin = (0, 0)
            if tag_prefix == "src":
                for pt in points_coords_array:
                    canvas.create_line(origin[0], origin[1], pt[0], pt[1], fill="darkgreen", width=1, arrow=tk.LAST,
                                       tags=f"{tag_prefix}_vectors")
            elif tag_prefix == "dst" and self.H is not None:
                # For destination vectors, transform the *source* points using H
                current_src_pts = self._get_points_from_entries(self.src_points_entries)
                if current_src_pts is None: return  # Need valid source points

                # Transform origin for destination vectors
                origin_homogeneous = np.array([0, 0, 1], dtype=np.float32)
                transformed_origin_homogeneous = self.H @ origin_homogeneous

                transformed_origin_pt = (float('inf'), float('inf'))  # Default if division by zero
                if abs(transformed_origin_homogeneous[2]) > 1e-6:
                    transformed_origin_pt = (transformed_origin_homogeneous[0] / transformed_origin_homogeneous[2],
                                             transformed_origin_homogeneous[1] / transformed_origin_homogeneous[2])

                for pt_src in current_src_pts:
                    pt_src_homogeneous = np.array([pt_src[0], pt_src[1], 1], dtype=np.float32)
                    transformed_pt_homogeneous = self.H @ pt_src_homogeneous

                    transformed_pt = (float('inf'), float('inf'))  # Default
                    if abs(transformed_pt_homogeneous[2]) > 1e-6:
                        transformed_pt = (transformed_pt_homogeneous[0] / transformed_pt_homogeneous[2],
                                          transformed_pt_homogeneous[1] / transformed_pt_homogeneous[2])

                    # Draw transformed vector if within reasonable bounds
                    if all(abs(c) < self.canvas_width * 10 for c in transformed_origin_pt) and \
                            all(abs(c) < self.canvas_height * 10 for c in transformed_pt):  # Generous bounds
                        canvas.create_line(transformed_origin_pt[0], transformed_origin_pt[1],
                                           transformed_pt[0], transformed_pt[1],
                                           fill="purple", width=1, arrow=tk.LAST, tags=f"{tag_prefix}_vectors")

    def _draw_initial_canvases(self):
        # This method just puts the base grid image on the canvases.
        # It does not draw any points or polygons initially.
        src_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil)
        self.source_canvas.create_image(0, 0, anchor=tk.NW, image=src_img_tk, tags="src_bg")
        self.source_canvas.image = src_img_tk

        dst_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil.copy())
        self.destination_canvas.create_image(0, 0, anchor=tk.NW, image=dst_img_tk, tags="dst_bg")
        self.destination_canvas.image = dst_img_tk

    def _update_3d_plot(self, src_pts_2d_array, dst_pts_2d_to_plot_array):
        self.ax_3d.clear()
        line_extension_factor = 2.0
        num_current_points = 0
        if src_pts_2d_array is not None:
            num_current_points = len(src_pts_2d_array)

        if src_pts_2d_array is not None and dst_pts_2d_to_plot_array is not None and \
                num_current_points >= 3 and len(dst_pts_2d_to_plot_array) == num_current_points:

            src_pts_3d = np.concatenate([src_pts_2d_array, np.zeros((num_current_points, 1))], axis=1)
            dst_pts_3d = np.concatenate([dst_pts_2d_to_plot_array, np.ones((num_current_points, 1))], axis=1)

            verts_src = [list(zip(src_pts_3d[:, 0], src_pts_3d[:, 1], src_pts_3d[:, 2]))]
            poly_src = Poly3DCollection(verts_src, alpha=0.6, facecolors='cyan', linewidths=1.5, edgecolors='blue')
            self.ax_3d.add_collection3d(poly_src)

            verts_dst = [list(zip(dst_pts_3d[:, 0], dst_pts_3d[:, 1], dst_pts_3d[:, 2]))]
            poly_dst = Poly3DCollection(verts_dst, alpha=0.6, facecolors='lightgreen', linewidths=1.5,
                                        edgecolors='green')
            self.ax_3d.add_collection3d(poly_dst)

            for i in range(num_current_points):
                s_pt = src_pts_3d[i]
                d_pt = dst_pts_3d[i]
                direction_vector = d_pt - s_pt

                line_start_pt = s_pt - line_extension_factor * direction_vector
                line_end_pt = d_pt + line_extension_factor * direction_vector

                self.ax_3d.plot([line_start_pt[0], line_end_pt[0]],
                                [line_start_pt[1], line_end_pt[1]],
                                [line_start_pt[2], line_end_pt[2]], 'r--', linewidth=1)

            all_x = np.concatenate([src_pts_3d[:, 0], dst_pts_3d[:, 0]])
            all_y = np.concatenate([src_pts_3d[:, 1], dst_pts_3d[:, 1]])
            # Add a small buffer or ensure min/max range for empty/single point cases if they were allowed
            min_x, max_x = (np.min(all_x) if all_x.size > 0 else 0) - self.grid_step, \
                           (np.max(all_x) if all_x.size > 0 else self.canvas_width) + self.grid_step
            min_y, max_y = (np.min(all_y) if all_y.size > 0 else 0) - self.grid_step, \
                           (np.max(all_y) if all_y.size > 0 else self.canvas_height) + self.grid_step

            self.ax_3d.set_xlim([min(0, min_x), max(self.canvas_width, max_x)])
            self.ax_3d.set_ylim([min(0, min_y), max(self.canvas_height, max_y)])
            self.ax_3d.set_zlim([-3, 3])
            self.ax_3d.set_xlabel('X');
            self.ax_3d.set_ylabel('Y');
            self.ax_3d.set_zlabel('Z (Conceptual Plane)')
            self.ax_3d.set_title('3D Plane-to-Plane Mapping (Extended Rays)')
        else:
            self.ax_3d.set_title('3D Visualization (Need >=3 valid points)');
            self.ax_3d.set_xlabel('X');
            self.ax_3d.set_ylabel('Y');
            self.ax_3d.set_zlabel('Z')
            self.ax_3d.set_xlim([0, self.canvas_width]);
            self.ax_3d.set_ylim([0, self.canvas_height]);
            self.ax_3d.set_zlim([-3, 3])
        self.canvas_3d_widget.draw()

    def _trigger_redraw_visuals(self):
        src_pts_current = self._get_points_from_entries(self.src_points_entries)
        dst_pts_current_ui = self._get_points_from_entries(self.dst_points_entries)

        if src_pts_current is None:
            self._draw_initial_canvases()
            self.source_canvas.delete("src_poly");
            self.source_canvas.delete("src_points");
            self.source_canvas.delete("src_vectors")
            self.destination_canvas.delete("dst_poly");
            self.destination_canvas.delete("dst_points");
            self.destination_canvas.delete("dst_vectors")
            self._update_3d_plot(None, None)
            return

        src_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil)
        self._draw_quad_and_points(self.source_canvas, src_pts_current, src_img_tk, color="blue", point_color="red",
                                   tag_prefix="src")

        if self.H is not None:
            warped_grid_cv = cv2.warpPerspective(self.base_grid_image_cv, self.H,
                                                 (self.canvas_width, self.canvas_height))
            warped_grid_pil = self._cv2_to_pil(warped_grid_cv)
            dst_img_tk_for_canvas = ImageTk.PhotoImage(warped_grid_pil)
        else:
            dst_img_tk_for_canvas = ImageTk.PhotoImage(self.base_grid_image_pil.copy())

        self._draw_quad_and_points(self.destination_canvas, dst_pts_current_ui, dst_img_tk_for_canvas, color="green",
                                   point_color="orange", tag_prefix="dst")

        self._update_3d_plot(src_pts_current, dst_pts_current_ui)

    def _on_compute_homography_from_points(self):
        src_pts = self._get_points_from_entries(self.src_points_entries)
        dst_pts_ui = self._get_points_from_entries(self.dst_points_entries)

        if src_pts is None or dst_pts_ui is None:
            self.H = None
            self._update_homography_display()
            self._trigger_redraw_visuals()
            return

        current_num_points = self.num_points_var.get()
        if len(src_pts) < 4 or len(dst_pts_ui) < 4 or current_num_points < 4:
            messagebox.showerror("Input Error",
                                 "At least 4 source and 4 destination points are required to compute Homography.")
            self.H = None
            self._update_homography_display()
            self._trigger_redraw_visuals()
            return

        if len(src_pts) != len(dst_pts_ui):  # Should be guaranteed by how entries are created/read
            messagebox.showerror("Input Error", "The number of source and destination points must match.")
            self.H = None
            self._update_homography_display()
            self._trigger_redraw_visuals()
            return

        H_matrix_computed, mask = cv2.findHomography(src_pts, dst_pts_ui, 0)

        if H_matrix_computed is None:
            messagebox.showerror("Error",
                                 "Could not compute Homography from points. Check if points are collinear or too few unique points.")
            self.H = None
        else:
            self.H = H_matrix_computed

        self._update_homography_display()
        self._trigger_redraw_visuals()

    def _on_apply_manual_homography(self):
        manual_H_values = []
        try:
            for r in range(3):
                for c in range(3):
                    manual_H_values.append(float(self.homography_matrix_entries[r][c].get()))
            current_H_from_ui = np.array(manual_H_values).reshape((3, 3))
        except ValueError:
            messagebox.showerror("Input Error", "Invalid numeric value in Homography Matrix entries.")
            return

        self.H = current_H_from_ui

        src_pts = self._get_points_from_entries(self.src_points_entries)
        if src_pts is None:
            messagebox.showerror("Input Error", "Source points are invalid. Cannot apply manual H.")
            self._trigger_redraw_visuals()  # Redraw with current H (manual) but possibly no points
            return

        new_dst_pts_calculated_list = []
        for pt_s in src_pts:
            s_homogeneous = np.array([pt_s[0], pt_s[1], 1], dtype=np.float32)
            d_homogeneous = self.H @ s_homogeneous

            if abs(d_homogeneous[2]) < 1e-6:
                new_dst_pts_calculated_list.append((float('inf'), float('inf')))
            else:
                new_dst_pts_calculated_list.append(
                    (d_homogeneous[0] / d_homogeneous[2], d_homogeneous[1] / d_homogeneous[2]))

        new_dst_pts_calculated_np_array = np.array(new_dst_pts_calculated_list, dtype=np.float32)

        self._update_point_entries(self.dst_points_entries, new_dst_pts_calculated_np_array)

        self._trigger_redraw_visuals()


if __name__ == "__main__":
    root = tk.Tk()
    app = HomographyVisualizerApp(root)
    root.mainloop()