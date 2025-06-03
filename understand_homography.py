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
        master.geometry("1000x950")  # Adjusted window size for new button

        self.canvas_width = 280
        self.canvas_height = 280
        self.grid_step = 20
        self.point_radius = 4
        self.line_width = 2

        self.src_points_entries = []
        self.dst_points_entries = []
        self.homography_matrix_entries = []  # Changed from labels to entries
        self.H = np.eye(3)  # Initialize H to identity

        self.base_grid_image_pil = self._create_grid_pil(self.canvas_width, self.canvas_height, self.grid_step)
        self.base_grid_image_cv = self._pil_to_cv2(self.base_grid_image_pil.copy())
        self.show_vectors_var = tk.BooleanVar(value=False)
        self._setup_ui()
        self._update_homography_display()  # Populate matrix entries with initial H
        self._draw_initial_canvases()

        # Initial computation to populate everything based on default points
        # This will also call _update_3d_plot indirectly
        self._on_compute_homography_from_points()


    def _pil_to_cv2(self, pil_image):
        open_cv_image = np.array(pil_image)
        if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
            open_cv_image = open_cv_image[:, :, ::-1].copy()
        elif len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 4:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGRA)
        return open_cv_image

    def _cv2_to_pil(self, cv_image):
        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
        else:
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

        src_frame = ttk.LabelFrame(left_panel, text="Source Points (x, y)", padding="10")
        src_frame.pack(pady=5, fill=tk.X)
        default_src_pts = [(50, 50), (200, 50), (200, 200), (50, 200)]
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

        dst_frame = ttk.LabelFrame(left_panel, text="Destination Points (x', y')", padding="10")
        dst_frame.pack(pady=5, fill=tk.X)
        default_dst_pts = [(70, 70), (230, 60), (240, 180), (60, 190)]
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
                entry_h.insert(0, f"{self.H[r, c]:.4f}")
                entry_h.pack(side=tk.LEFT, padx=2, pady=2)
                row_entries.append(entry_h)
            self.homography_matrix_entries.append(row_entries)

        apply_h_button = ttk.Button(left_panel, text="Apply Manual H & Update Dst Pts",
                                    command=self._on_apply_manual_homography)
        apply_h_button.pack(pady=(0, 10))

        right_panel = ttk.Frame(main_frame, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        top_right_panel = ttk.Frame(right_panel)
        top_right_panel.pack(fill=tk.X, expand=False)

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
        bottom_right_panel.pack(fill=tk.BOTH, expand=True, pady=10)

        self.fig_3d = Figure(figsize=(6, 5), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d_widget = FigureCanvasTkAgg(self.fig_3d, master=bottom_right_panel)
        self.canvas_3d_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _get_points_from_entries(self, point_entries):
        """Helper to get points from a list of (entry_x, entry_y) tuples."""
        pts = []
        try:
            for ex, ey in point_entries:
                pts.append((float(ex.get()), float(ey.get())))
        except ValueError:
            messagebox.showerror("Input Error", "Invalid numeric coordinates in point entries.")
            return None
        if len(pts) != 4:  # Should always be 4 based on setup
            messagebox.showerror("Input Error", "Incorrect number of point entries provided.")
            return None
        return np.array(pts, dtype=np.float32)

    def _update_point_entries(self, point_entries_ui, points_data):
        """Updates UI point entries with new data."""
        if points_data is None or len(points_data) != 4:
            return
        for i in range(4):
            ex, ey = point_entries_ui[i]
            ex.delete(0, tk.END)
            ex.insert(0, f"{points_data[i, 0]:.2f}")
            ey.delete(0, tk.END)
            ey.insert(0, f"{points_data[i, 1]:.2f}")

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

    def _draw_quad_and_points(self, canvas, points_coords, tk_image_bg, color="blue", point_color="red", tag_prefix=""):
        canvas.delete(f"{tag_prefix}_quad")
        canvas.delete(f"{tag_prefix}_points")
        canvas.delete(f"{tag_prefix}_vectors")
        canvas.delete(f"{tag_prefix}_bg")

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
                # This part needs src_pts to transform, not dst_pts.
                # We should use the current src_pts from UI entries for this.
                current_src_pts = self._get_points_from_entries(self.src_points_entries)
                if current_src_pts is None: return

                origin_homogeneous = np.array([0, 0, 1], dtype=np.float32)
                transformed_origin_homogeneous = self.H @ origin_homogeneous

                transformed_origin_pt = (float('inf'), float('inf'))
                if abs(transformed_origin_homogeneous[2]) > 1e-6:
                    transformed_origin_pt = (transformed_origin_homogeneous[0] / transformed_origin_homogeneous[2],
                                             transformed_origin_homogeneous[1] / transformed_origin_homogeneous[2])

                for pt_src in current_src_pts:
                    pt_src_homogeneous = np.array([pt_src[0], pt_src[1], 1], dtype=np.float32)
                    transformed_pt_homogeneous = self.H @ pt_src_homogeneous

                    transformed_pt = (float('inf'), float('inf'))
                    if abs(transformed_pt_homogeneous[2]) > 1e-6:
                        transformed_pt = (transformed_pt_homogeneous[0] / transformed_pt_homogeneous[2],
                                          transformed_pt_homogeneous[1] / transformed_pt_homogeneous[2])

                    if all(abs(c) < self.canvas_width * 5 for c in transformed_origin_pt) and \
                            all(abs(c) < self.canvas_height * 5 for c in transformed_pt):
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

    def _update_3d_plot(self, src_pts_2d, dst_pts_2d_to_plot):
        self.ax_3d.clear()
        if src_pts_2d is not None and dst_pts_2d_to_plot is not None and \
                len(src_pts_2d) == 4 and len(dst_pts_2d_to_plot) == 4:
            src_pts_3d = np.concatenate([src_pts_2d, np.zeros((4, 1))], axis=1)
            dst_pts_3d = np.concatenate([dst_pts_2d_to_plot, np.ones((4, 1))], axis=1)

            verts_src = [
                list(zip(src_pts_3d[[0, 1, 2, 3], 0], src_pts_3d[[0, 1, 2, 3], 1], src_pts_3d[[0, 1, 2, 3], 2]))]
            poly_src = Poly3DCollection(verts_src, alpha=0.6, facecolors='cyan', linewidths=1.5, edgecolors='blue')
            self.ax_3d.add_collection3d(poly_src)

            verts_dst = [
                list(zip(dst_pts_3d[[0, 1, 2, 3], 0], dst_pts_3d[[0, 1, 2, 3], 1], dst_pts_3d[[0, 1, 2, 3], 2]))]
            poly_dst = Poly3DCollection(verts_dst, alpha=0.6, facecolors='lightgreen', linewidths=1.5,
                                        edgecolors='green')
            self.ax_3d.add_collection3d(poly_dst)

            for i in range(4):
                self.ax_3d.plot([src_pts_3d[i, 0], dst_pts_3d[i, 0]],
                                [src_pts_3d[i, 1], dst_pts_3d[i, 1]],
                                [src_pts_3d[i, 2], dst_pts_3d[i, 2]], 'r--', linewidth=1)

            all_x = np.concatenate([src_pts_3d[:, 0], dst_pts_3d[:, 0]])
            all_y = np.concatenate([src_pts_3d[:, 1], dst_pts_3d[:, 1]])
            min_x, max_x = np.min(all_x) - self.grid_step, np.max(all_x) + self.grid_step
            min_y, max_y = np.min(all_y) - self.grid_step, np.max(all_y) + self.grid_step

            self.ax_3d.set_xlim([min(0, min_x), max(self.canvas_width, max_x)])
            self.ax_3d.set_ylim([min(0, min_y), max(self.canvas_height, max_y)])
            self.ax_3d.set_zlim([-0.5, 1.5])
            self.ax_3d.set_xlabel('X');
            self.ax_3d.set_ylabel('Y');
            self.ax_3d.set_zlabel('Z (Conceptual Plane)')
            self.ax_3d.set_title('3D Plane-to-Plane Mapping')
        else:
            self.ax_3d.set_title('3D Visualization (Enter valid points)');
            self.ax_3d.set_xlabel('X');
            self.ax_3d.set_ylabel('Y');
            self.ax_3d.set_zlabel('Z')
            self.ax_3d.set_xlim([0, self.canvas_width]);
            self.ax_3d.set_ylim([0, self.canvas_height]);
            self.ax_3d.set_zlim([-0.5, 1.5])
        self.canvas_3d_widget.draw()

    def _trigger_redraw_visuals(self):
        """Redraws visuals, typically used by checkbox or when only H changes."""
        src_pts = self._get_points_from_entries(self.src_points_entries)
        dst_pts_ui = self._get_points_from_entries(self.dst_points_entries)  # Get current UI dst points

        if src_pts is None:  # If src_pts are invalid, clear and return
            self._draw_initial_canvases()
            self._update_3d_plot(None, None)
            return

        # Update Source Canvas
        src_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil)
        self._draw_quad_and_points(self.source_canvas, src_pts, src_img_tk, color="blue", point_color="red",
                                   tag_prefix="src")

        # Update Destination Canvas
        # If H is None (e.g. after an error), draw plain grid with dst_pts_ui
        # Otherwise, use self.H to warp grid and draw dst_pts_ui
        if self.H is not None:
            warped_grid_cv = cv2.warpPerspective(self.base_grid_image_cv, self.H,
                                                 (self.canvas_width, self.canvas_height))
            warped_grid_pil = self._cv2_to_pil(warped_grid_cv)
            dst_img_tk = ImageTk.PhotoImage(warped_grid_pil)
        else:  # H is None, use plain grid
            dst_img_tk = ImageTk.PhotoImage(self.base_grid_image_pil.copy())

        # dst_pts_ui are the points currently in the destination UI fields. These are used for drawing the quad.
        self._draw_quad_and_points(self.destination_canvas, dst_pts_ui, dst_img_tk, color="green", point_color="orange",
                                   tag_prefix="dst")

        self._update_3d_plot(src_pts, dst_pts_ui)

    def _on_compute_homography_from_points(self):
        src_pts = self._get_points_from_entries(self.src_points_entries)
        dst_pts_ui = self._get_points_from_entries(self.dst_points_entries)

        if src_pts is None or dst_pts_ui is None:
            self.H = None
            self._update_homography_display()
            self._trigger_redraw_visuals()  # Redraw with available points or clear
            return

        H_matrix_computed, mask = cv2.findHomography(src_pts, dst_pts_ui, 0)

        if H_matrix_computed is None:
            messagebox.showerror("Error",
                                 "Could not compute Homography from points. Check if points are collinear or too few unique points.")
            self.H = None
        else:
            self.H = H_matrix_computed

        self._update_homography_display()  # Update H matrix entries with computed H
        self._trigger_redraw_visuals()  # Redraw all visuals based on new H and current UI points

    def _on_apply_manual_homography(self):
        manual_H_values = []
        try:
            for r in range(3):
                for c in range(3):
                    manual_H_values.append(float(self.homography_matrix_entries[r][c].get()))
            self.H = np.array(manual_H_values).reshape((3, 3))
        except ValueError:
            messagebox.showerror("Input Error", "Invalid numeric value in Homography Matrix entries.")
            self.H = None  # Invalidate H
            self._update_homography_display()  # Show N/A or revert
            return

        src_pts = self._get_points_from_entries(self.src_points_entries)
        if src_pts is None:
            messagebox.showerror("Input Error", "Source points are invalid. Cannot apply manual H.")
            # Potentially clear dst point entries or leave them
            return

        # Calculate new destination points based on manual H and src_pts
        new_dst_pts_calculated = []
        for pt_s in src_pts:
            s_homogeneous = np.array([pt_s[0], pt_s[1], 1], dtype=np.float32)
            d_homogeneous = self.H @ s_homogeneous

            if abs(d_homogeneous[2]) < 1e-6:  # Point at or near infinity
                # messagebox.showwarning("Calculation Warning", f"Source point {pt_s} maps to infinity with this H.")
                # Use a very large coordinate or a placeholder, or skip. For now, inf.
                new_dst_pts_calculated.append((float('inf'), float('inf')))
            else:
                new_dst_pts_calculated.append(
                    (d_homogeneous[0] / d_homogeneous[2], d_homogeneous[1] / d_homogeneous[2]))

        new_dst_pts_calculated_np = np.array(new_dst_pts_calculated, dtype=np.float32)

        # Update the UI destination point entries with these new_dst_pts_calculated
        self._update_point_entries(self.dst_points_entries, new_dst_pts_calculated_np)

        # Redraw visuals. _trigger_redraw_visuals will use the updated H and current UI points (which now reflect new_dst_pts_calculated).
        self._trigger_redraw_visuals()


if __name__ == "__main__":
    root = tk.Tk()
    app = HomographyVisualizerApp(root)
    root.mainloop()