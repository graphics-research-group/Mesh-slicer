#include <iostream>
#include <fstream>
#include <limits.h>
#include <filesystem>
#include <time.h>

// UI
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/ViewerData.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/file_dialog_open.h>

// Mesh
#include <igl/read_triangle_mesh.h>
#include <igl/unproject_onto_mesh.h>

// Skeletonization
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <igl/copyleft/cgal/mesh_to_polyhedron.h>
#include "unproject_on_skeleton.h"

// Surface fitting and intersection
#include <igl/unproject.h>
#include <igl/writeOBJ.h>
#include <igl/copyleft/cgal/trim_with_solid.h>
#include <igl/boundary_loop.h>
#include <igl/slice_into.h>
#include <igl/slice.h>
#include <igl/colon.h>
#include <igl/random_points_on_mesh.h>
#include <igl/copyleft/cgal/points_inside_component.h>
#include <igl/copyleft/cgal/wire_mesh.h>
#include <igl/bounding_box_diagonal.h>

#include "fitsurface.h"
#include "defines.h"

#define MAX_SELECTED_POINTS 100

void init();
void clearPickedPoints(igl::opengl::glfw::Viewer &viewer);
void populateMenu(igl::opengl::glfw::imgui::ImGuiMenu &menu, igl::opengl::glfw::Viewer &viewer);
void computeSkeleton(Eigen::MatrixXd &mV, Eigen::MatrixXi &mF, Eigen::MatrixXd &sV, Eigen::MatrixXi &sE);
bool fitCuttingSurface(FittingType ftype);
bool pre_draw(igl::opengl::glfw::Viewer & viewer);
void setMouseCallback(igl::opengl::glfw::Viewer & viewer);
void setKeyboardCallback(igl::opengl::glfw::Viewer & viewer);
void saveCuttingSurfaces();
void intersectMeshMesh(Eigen::MatrixXd &V_solid, Eigen::MatrixXi &F_solid, Eigen::MatrixXd &V_surf, Eigen::MatrixXi &F_surf, Eigen::MatrixXd &V_surfReg, Eigen::VectorXi &V_surfRegIndicator, std::vector<std::vector<int> > &contours, Eigen::MatrixXd &V_Cut, Eigen::MatrixXi &F_Cut, Eigen::MatrixXi &F_Cut_in, Eigen::MatrixXi &F_Cut_out);
void contourChainsToEdges(const std::vector<std::vector<int> > &chains, Eigen::MatrixXi &E);
void sampleRandomPoints(const Eigen::MatrixXd &meshV, const Eigen::MatrixXi &meshF, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, size_t nsamples, Eigen::MatrixXd &samples, Eigen::VectorXi &samplesIndicator);
void sampleVicinityPoints(const Eigen::MatrixXd &meshV, const Eigen::MatrixXi &meshF, const Eigen::MatrixXd &surfV, const Eigen::MatrixXi &surfF, const Eigen::MatrixXd &contV, const Eigen::MatrixXi &contE, Eigen::MatrixXd &vicinityV, Eigen::MatrixXi &vicinityIndicator);
void resampleContour(double delta, std::vector<int> contour, const Eigen::MatrixXd &points, std::vector<int> &recontour, Eigen::MatrixXd &repoints);

struct CuttingSurface {
  FittingType surfaceType;
  Eigen::MatrixXd worldToLocal; // 4x4 PCA matrix
  Eigen::VectorXd coefficients; // Fitting coefficients
  Eigen::MatrixXd mesh_pickedPoints; // Mesh points
  Eigen::MatrixXd skel_pickedPoints; // Skeleton points
  Eigen::MatrixXd surface_V; // Surface vertices
  Eigen::MatrixXi surface_F; // Surface faces
  Eigen::MatrixXd surfaceReg_V; // Regular surface points
  Eigen::MatrixXd surfaceRand_V; // Random surface points
  Eigen::VectorXi surfaceReg_V_indicator;// Indicator function {-1, 0, 1}
  Eigen::VectorXi surfaceRand_V_indicator;// Indicator function {-1, 0, 1}
  std::vector<std::vector<int> > contours; // cross-section Countours
  Eigen::MatrixXd contour_V;// Contour vertices
  Eigen::MatrixXi contour_E;// Contour edges (derived from 'contours')
  Eigen::MatrixXd surfaceCut_V; // Intersection of cutting surface and mesh solid
  Eigen::MatrixXi surfaceCut_F; // Faces of cut surface
  Eigen::MatrixXi surfaceCut_F_in; // Inside faces of cut surface
  Eigen::MatrixXi surfaceCut_F_out; // Outside faces of cut surface
  Eigen::MatrixXd contourVicinity_V;// Points sampled in the vicinity of a contour
  Eigen::MatrixXi contourVicinity_V_indicator;// Indicator function {-1, 0, 1}}
};

struct Scene {
  // Objects
  Eigen::MatrixXd mesh_V; // Mesh vertices
  Eigen::MatrixXi mesh_F; // Mesh faces
  Eigen::MatrixXd skel_V; // Skeleton vertices
  Eigen::MatrixXi skel_E; // Skeleton faces
  Eigen::MatrixXd mesh_pickedPoints;
  Eigen::MatrixXd skel_pickedPoints;
  int mesh_nPicked, skel_nPicked;
  std::vector<CuttingSurface> cuttingSurfaces;
};

//Global vars
Scene scene;
std::string mesh_filename;
Eigen::RowVector3d mesh_color(255./255.,216./255.,1./255.);//Yellow
Eigen::RowVector3d skel_color(70./255.,252./255.,167./255.); //Sea-green
float skel_width;
Eigen::RowVector3d mesh_pickedColor(255./255., 0./255., 0./255.); //Red
Eigen::RowVector3d skel_pickedColor(0./255., 0./255., 255./255.); //Blue
int key_modifier;
int cutting_surface_min_res;
int selected_cutting_surface;
Eigen::RowVector3d xsection_color(0.1, 0.9, 0.1);
Eigen::RowVector3d xsection_selectedColor(0.9, 0.1, 0.1);
Eigen::RowVector3d contour_color(0.8, 0.3, 0.1);
bool erase_points_after_surface_fitting; // Caution: when set to false, it is possible to accidently create identical cutting surfaces
bool display_all_cutting_surface_meshes; 
bool display_all_cutting_surface_seeds;
int num_random_points_on_cutting_surface; 
float vicinity_sample_percent_radius;
float vicinity_sample_percent_delta;
bool sampled_all_points;
bool resample_contour;
float resample_contour_percent_delta;

//Note: The IGL viewer contains data in following fashion:
// 1. The Mesh, skeleton and currently selected points are stored in viewer.data(0)
// 2. Cutting surfaces and their corresponding selected points are stored in viewer.data(k), where k>0

int main(int argc, char *argv[])
{
  //Intialize data structures, allocate memory, etc.
  init();

  // Initialize viewer
  igl::opengl::glfw::Viewer viewer;
  std::cout<<"Shift + click\tPick points on mesh\n";
  std::cout<<"Control + click\tPick points on skeleton\n";
  std::cout<<"Alt + click\tSelect cutting surface\n";

  // Attach menu
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  // Populate menu
  populateMenu(menu, viewer);

  // Set draw properties
  viewer.data(0).show_lines = false;
  viewer.data(0).show_overlay_depth = false; // Show skeleton as an overlay, discarding depth.
  viewer.data(0).line_width = skel_width;
  viewer.data(0).point_size = 10.0;

  // Set callbacks
  setMouseCallback(viewer);
  setKeyboardCallback(viewer);
  viewer.callback_pre_draw = &pre_draw;

  // Launch window
  viewer.launch();
}

// Initialize data in this scope
void init()
{
  scene.mesh_pickedPoints.resize(MAX_SELECTED_POINTS, 3);
  scene.skel_pickedPoints.resize(MAX_SELECTED_POINTS, 3);
  scene.mesh_nPicked = 0;
  scene.skel_nPicked = 0;
  key_modifier = 0;
  skel_width = 1;
  cutting_surface_min_res = 20;
  selected_cutting_surface=-1; // Use to index idx into view.data(idx), and scene.cuttingSurfaces[idx -1]
  erase_points_after_surface_fitting = true;
  display_all_cutting_surface_meshes = false; 
  display_all_cutting_surface_seeds = false;
  num_random_points_on_cutting_surface = 1000;
  vicinity_sample_percent_radius = 0.2;
  vicinity_sample_percent_delta = 0.2;
  sampled_all_points = false;
  resample_contour = true;
  resample_contour_percent_delta = 0.2;
}

// Clear data in this scope
void clearPickedPoints(igl::opengl::glfw::Viewer &viewer)
{
  scene.mesh_pickedPoints.setZero();
  scene.skel_pickedPoints.setZero();
  scene.mesh_nPicked = 0;
  scene.skel_nPicked = 0;
  key_modifier = 0;
  viewer.data(0).clear_points();
}

void setMouseCallback(igl::opengl::glfw::Viewer & viewer)
{
  viewer.callback_mouse_down = 
  [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
  {
    if(key_modifier & IGL_MOD_SHIFT) //Select points on Mesh
    {
      int fid;
      Eigen::Vector3d bc;
      Eigen::RowVector3d pt;
      // Cast a ray in the view direction starting from the mouse position
      double x = viewer.current_mouse_x;
      double y = viewer.core().viewport(3) - viewer.current_mouse_y;
      if(igl::unproject_onto_mesh(
                Eigen::Vector2f(x,y),
                viewer.core().view,
                viewer.core().proj,
                viewer.core().viewport,
                scene.mesh_V,
                scene.mesh_F,
                fid,
                bc))
      {
        pt =    bc[0]*scene.mesh_V.row(scene.mesh_F(fid,0)) + 
                bc[1]*scene.mesh_V.row(scene.mesh_F(fid,1)) + 
                bc[2]*scene.mesh_V.row(scene.mesh_F(fid,2));       
        scene.mesh_pickedPoints.row(scene.mesh_nPicked++) = pt; // Append to the end
        viewer.data(0).add_points(scene.mesh_pickedPoints.block(0, 0, scene.mesh_nPicked, 3), mesh_pickedColor);
        return true;
      }
    }
    if(key_modifier & IGL_MOD_CONTROL) //Select points on Skeleton
    {
      int fid;
      Eigen::RowVector3d pt;
      // Cast a ray in the view direction starting from the mouse position
      double x = viewer.current_mouse_x;
      double y = viewer.core().viewport(3) - viewer.current_mouse_y;
      if(unproject_on_skeleton(
                Eigen::Vector2f(x,y),
                viewer.core().proj * viewer.core().view,
                viewer.core().viewport,
                scene.skel_V,
                scene.skel_E,
                pt))
      {
        scene.skel_pickedPoints.row(scene.skel_nPicked++) = pt; // Append to the end
        viewer.data(0).add_points(scene.skel_pickedPoints.block(0, 0, scene.skel_nPicked, 3), skel_pickedColor);
        return true;
      }
    }
    if(key_modifier & IGL_MOD_ALT) //Select a cutting surface
    {
      int nsurfs = viewer.data_list.size() - 1;
      if(nsurfs > 0) {
        int fid;
        Eigen::Vector3d bc;
        Eigen::RowVector3d pt;
        std::vector<double> dists(nsurfs, DBL_MAX);
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        Eigen::RowVector3d win_pos(x, y, 0);
        Eigen::RowVector3d scene_pos;
        igl::unproject(win_pos, viewer.core().view, viewer.core().proj, viewer.core().viewport, scene_pos);
        bool intersected = false;
        for(int i = 0; i < nsurfs; i++) {
          Eigen::MatrixXd &surf_V = scene.cuttingSurfaces[i].surface_V;
          Eigen::MatrixXi &surf_F = scene.cuttingSurfaces[i].surface_F;
          // Cast a ray in the view direction starting from the mouse position
          if(igl::unproject_onto_mesh(
                      Eigen::Vector2f(x,y),
                      viewer.core().view,
                      viewer.core().proj,
                      viewer.core().viewport,
                      surf_V,
                      surf_F,
                      fid,
                      bc))
          {
            pt = bc[0]*surf_V.row(surf_F(fid,0)) + 
                bc[1]*surf_V.row(surf_F(fid,1)) + 
                bc[2]*surf_V.row(surf_F(fid,2));       
            dists[i] = (scene_pos - pt).norm();
            intersected = true;
          }
          if(!intersected)
            selected_cutting_surface = -1;
          else
            selected_cutting_surface = std::min_element(dists.begin(), dists.end()) - dists.begin() + 1;
        }
      }
      return true;
    }
    return false;

  };
}

void setKeyboardCallback(igl::opengl::glfw::Viewer & viewer)
{
  viewer.callback_key_down = 
  [&](igl::opengl::glfw::Viewer & viewer, unsigned char key, int _mod)->bool
  {
    key_modifier = _mod;
    return false;
  };

  viewer.callback_key_up = 
  [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int _mod)->bool
  {
    key_modifier = _mod;
    return false;
  };
}

bool pre_draw(igl::opengl::glfw::Viewer & viewer)
{
  int nmeshes = viewer.data_list.size();
  if( nmeshes > 1) // Then highlight the mesh
  {
    for(int i=1; i<nmeshes; i++)
      viewer.data_list[i].set_colors(xsection_color);
    if(selected_cutting_surface >= 1 && selected_cutting_surface < nmeshes)
      viewer.data_list[selected_cutting_surface].set_colors(xsection_selectedColor);
  }
  return false;
}

void populateMenu(igl::opengl::glfw::imgui::ImGuiMenu &menu, igl::opengl::glfw::Viewer &viewer)
{
  menu.callback_draw_viewer_window = [&]()
  {
    // Define next window position + size
    ImGui::SetNextWindowPos(ImVec2(0.0f * menu.menu_scaling(), 0), ImGuiSetCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(200, 400), ImGuiSetCond_FirstUseEver);
    ImGui::Begin(
      "Main", nullptr,
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize  
    );

    //Mesh load and properties
    if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::Button("Load Mesh"))
      {
        mesh_filename = igl::file_dialog_open();
        bool result = igl::read_triangle_mesh(mesh_filename, scene.mesh_V, scene.mesh_F);
        if(result) {
          viewer.data(0).clear();
          clearPickedPoints(viewer);
          viewer.data(0).set_mesh(scene.mesh_V, scene.mesh_F);
          viewer.data(0).set_colors(mesh_color);
          viewer.core(0).align_camera_center(scene.mesh_V, scene.mesh_F);
          std::cout << "Read mesh "<< mesh_filename <<"\n";
          std::cout << "Mesh:\n\tVertices - "<<scene.mesh_V.rows()<<"\n\tFaces - "<<scene.mesh_F.rows()<<"\n";
        } 
      }
      static float ms_color[] = {(float)mesh_color[0], (float)mesh_color[1], (float)mesh_color[2]};
      if(ImGui::ColorEdit4("Mesh Face Color", ms_color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel))
      {
        mesh_color << (double)ms_color[0], (double)ms_color[1], (double)ms_color[2];
        viewer.data(0).set_colors(mesh_color);
      }
      ImGui::SameLine();
      ImGui::Checkbox("Fill", 
        [&]() { return viewer.data(0).show_faces;},
        [&](bool value) { return viewer.data(0).show_faces = value;}
        );
      ImGui::ColorEdit4("Mesh line Color", viewer.data(0).line_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
      ImGui::SameLine();
      ImGui::Checkbox("Wireframe", 
        [&]() { return viewer.data(0).show_lines;},
        [&](bool value) { return viewer.data(0).show_lines = value;}
        );
      ImGui::ColorEdit4("Background Color", viewer.core().background_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
      ImGui::SameLine();
      ImGui::Text("Background color");
      if (ImGui::Button("Center object"))
      {
        viewer.core().align_camera_center(viewer.data(0).V, viewer.data(0).F);
      }
      ImGui::SameLine();
      if (ImGui::Button("Snap view"))
      {
        viewer.snap_to_canonical_quaternion();
      }
    }
    
    // Skeleton computation and properties
    if (ImGui::CollapsingHeader("Skeleton", ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::Button("Compute Skeleton"))
      {
        computeSkeleton(scene.mesh_V, scene.mesh_F, scene.skel_V, scene.skel_E);
        viewer.data(0).clear_edges();
        viewer.data(0).set_edges(scene.skel_V, scene.skel_E, skel_color);
      }
      static float sk_color[] = {(float)skel_color[0], (float)skel_color[1], (float)skel_color[2]};
      if(ImGui::ColorEdit4("Skeleton Color", sk_color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel))
      {
        skel_color << (double)sk_color[0], (double)sk_color[1], (double)sk_color[2];
        viewer.data(0).set_edges(scene.skel_V, scene.skel_E, skel_color);
      }
      ImGui::SameLine();
      ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.20f);
      if(ImGui::DragFloat("Width", &skel_width, 0.1, 0.1, 5.0, "%.1f"))
      {
        viewer.data(0).line_width = skel_width; // Doesn't work on Apple/Windows
      }
    }

    // Seed points for surface generation
    if (ImGui::CollapsingHeader("Seed points", ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(ImGui::Button("Clear all points"))
      {
	clearPickedPoints(viewer);
      }

      ImGui::Text("Object");
      ImGui::SameLine();
      if(ImGui::Button("Clear points##Mesh"))
      {
        scene.mesh_pickedPoints.setZero();
        scene.mesh_nPicked = 0;
        viewer.data(0).clear_points();
        if(scene.skel_nPicked > 0)
          viewer.data(0).add_points(scene.skel_pickedPoints.block(0, 0, scene.skel_nPicked, 3), skel_pickedColor);
      }
      ImGui::SameLine();
      if(ImGui::Button("Clear last point##Mesh"))
      {
        scene.mesh_nPicked--;
        if(scene.mesh_nPicked <=0) scene.mesh_nPicked = 0;
        viewer.data(0).clear_points();
        if(scene.mesh_nPicked > 0)
          viewer.data(0).add_points(scene.mesh_pickedPoints.block(0, 0, scene.mesh_nPicked, 3), mesh_pickedColor);
        if(scene.skel_nPicked > 0)
          viewer.data(0).add_points(scene.skel_pickedPoints.block(0, 0, scene.skel_nPicked, 3), skel_pickedColor);
      }

      ImGui::Text("Skeleton");
      ImGui::SameLine();
      if(ImGui::Button("Clear points##Skeleton"))
      {
        scene.skel_pickedPoints.setZero();
        scene.skel_nPicked = 0;
        viewer.data(0).clear_points();
        if(scene.mesh_nPicked > 0)
          viewer.data(0).add_points(scene.mesh_pickedPoints.block(0, 0, scene.mesh_nPicked, 3), mesh_pickedColor);
      }
      ImGui::SameLine();
      if(ImGui::Button("Clear last point##Skeleton"))
      {
        scene.skel_nPicked--;
        if(scene.skel_nPicked <=0) scene.skel_nPicked = 0;
        viewer.data(0).clear_points();
        if(scene.skel_nPicked > 0)
          viewer.data(0).add_points(scene.skel_pickedPoints.block(0, 0, scene.skel_nPicked, 3), skel_pickedColor);
        if(scene.mesh_nPicked > 0)
          viewer.data(0).add_points(scene.mesh_pickedPoints.block(0, 0, scene.mesh_nPicked, 3), mesh_pickedColor);
      }

    }
    
    // Surface fitting computation and properties
    if (ImGui::CollapsingHeader("Cutting surfaces", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.20f);
      ImGui::DragInt("Min res", &cutting_surface_min_res, 5, 5, 200,"%d");
      static int fittype = 0;
      if(ImGui::RadioButton("Plane", fittype == FittingType_Plane)) { fittype = FittingType_Plane; } ImGui::SameLine();
      if(ImGui::RadioButton("Poly33", fittype == FittingType_Poly33)) { fittype = FittingType_Poly33; } ImGui::SameLine();
      if(ImGui::RadioButton("Biharmonic", fittype == FittingType_Biharmonic)) { fittype = FittingType_Biharmonic; }

      ImGui::Checkbox("Resample contour", &resample_contour);
      ImGui::SameLine();
      ImGui::DragFloat("% BBox Delta##Contour", &resample_contour_percent_delta, 0.05, 0.1, 10.0, "%.2f");

      if(ImGui::Button("Fit Surface"))
      {
        bool result = fitCuttingSurface((FittingType)fittype);
        if(result) {
          // Add cutting mesh and seed points to new viewerData
          viewer.append_mesh();
          viewer.data().clear();
          CuttingSurface &surf = scene.cuttingSurfaces.back();
          if(display_all_cutting_surface_meshes) {
            viewer.data().set_mesh(surf.surface_V, surf.surface_F);
            viewer.data().set_colors(xsection_color);
          }
          
          // Add seed points
          if(display_all_cutting_surface_seeds) {
            if(surf.mesh_pickedPoints.rows() > 0)
              viewer.data().add_points(surf.mesh_pickedPoints, mesh_pickedColor);
            if(surf.skel_pickedPoints.rows() > 0)
              viewer.data().add_points(surf.skel_pickedPoints, skel_pickedColor);
          }
          
          // Resample and extract contours
          if(resample_contour) {
            float delta = (resample_contour_percent_delta/100.0)*igl::bounding_box_diagonal(scene.mesh_V);
            std::vector<Eigen::MatrixXd> recontourV_vec;
            for(int i=0; i<surf.contours.size(); i++) {
              std::vector<int> recontour;
              Eigen::MatrixXd recontourV;
              resampleContour(delta, surf.contours[i], surf.surfaceCut_V, recontour, recontourV); 
              surf.contours[i] = recontour;
              recontourV_vec.push_back(recontourV);
            }

            //Merge all resampled points
            if(recontourV_vec.size() > 0) {
              size_t currentRow = 0, vsize = 0;
              for(int i=0; i<recontourV_vec.size(); i++) vsize += recontourV_vec[i].rows();
              surf.contour_V.resize(vsize, 3);
              for(int i=0; i<recontourV_vec.size(); i++) {
                if(i>0) for (int & v : surf.contours[i]) v += currentRow;
                surf.contour_V.block(currentRow, 0, recontourV_vec[i].rows(), 3) = recontourV_vec[i];
                currentRow += recontourV_vec[i].rows();
              }
            } else 
              surf.contour_V = recontourV_vec[0];

            //Convert to edge array
            contourChainsToEdges(surf.contours, surf.contour_E);
          } else {
            Eigen::MatrixXi E, I;
            contourChainsToEdges(surf.contours, E);
            igl::remove_unreferenced(surf.surfaceCut_V, E, surf.contour_V, surf.contour_E, I);
          }

          //Display contours
          viewer.data().set_edges(surf.contour_V, surf.contour_E, contour_color);
          //viewer.data().set_points(surf.contour_V, contour_color);//Uncomment to display points of contour
          viewer.data().line_width = 2.0;
          //viewer.data().show_overlay_depth = false; // Perhaps we don't need this option for displaying contours.
          viewer.data().show_lines = true;
          viewer.data().point_size = 10.0;

          // Clear selected points
          if(erase_points_after_surface_fitting)
              clearPickedPoints(viewer);
          selected_cutting_surface = viewer.data_list.size()-1;
        }
      }
      ImGui::SameLine();
      ImGui::Checkbox("Erase points after fit", &erase_points_after_surface_fitting);
      if(ImGui::Checkbox("Display meshes", &display_all_cutting_surface_meshes)) {
        int n = viewer.data_list.size();
        if(n > 1)
          for(int i = 1; i < n; i++) {
            viewer.data(i).V.resize(0, 3);
            viewer.data(i).F.resize(0, 3);
            if(display_all_cutting_surface_meshes) {
              viewer.data(i).set_mesh(scene.cuttingSurfaces[i-1].surface_V, scene.cuttingSurfaces[i-1].surface_F);
              viewer.data(i).set_colors(xsection_color);
            }
          }
      }
      ImGui::SameLine();
      if(ImGui::Checkbox("Display seeds", &display_all_cutting_surface_seeds)) {
        int n = viewer.data_list.size();
        if(n > 1)
          for(int i = 1; i < n; i++) {
            viewer.data(i).clear_points();
            if(display_all_cutting_surface_seeds) {
              if(scene.cuttingSurfaces[i-1].mesh_pickedPoints.rows() > 0)
                viewer.data(i).add_points(scene.cuttingSurfaces[i-1].mesh_pickedPoints, mesh_pickedColor);
              if(scene.cuttingSurfaces[i-1].skel_pickedPoints.rows() > 0)
                viewer.data(i).add_points(scene.cuttingSurfaces[i-1].skel_pickedPoints, skel_pickedColor);
            }
          }
      }
      ImGui::SameLine();
      if(ImGui::Button("Show/Hide Reg. points"))
      {
        if((selected_cutting_surface > 0)) {
          size_t npts = viewer.data(selected_cutting_surface).points.rows();
          CuttingSurface &surface = scene.cuttingSurfaces.at(selected_cutting_surface-1);
          if(npts > 0) // Hide
            viewer.data(selected_cutting_surface).clear_points(); 
          else { // Show
            if(surface.surfaceReg_V.rows() > 0) {
              Eigen::MatrixXd C = Eigen::MatrixXd::Zero(surface.surfaceReg_V.rows(), 3);
                C.col(0) = surface.surfaceReg_V_indicator.unaryExpr([](int i){ return (i==MARKER_OUTSIDE)?1.0:0.0;});
                C.col(1) = surface.surfaceReg_V_indicator.unaryExpr([](int i){ return (i==MARKER_BOUNDARY)?1.0:0.0;});
                C.col(2) = surface.surfaceReg_V_indicator.unaryExpr([](int i){ return (i==MARKER_INSIDE)?1.0:0.0;});
              viewer.data(selected_cutting_surface).set_points(surface.surfaceReg_V, C); 
            }
          }
        }
      }

      ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
      if(ImGui::InputInt("Selected", &selected_cutting_surface))
      {
        // Cutting surfaces are numbered starting index 1 in the viewer.
        int n = viewer.data_list.size() - 1; // Number of cutting surfaces in the viewer
        if(n == 0)
          selected_cutting_surface = -1;
        else {
          if(selected_cutting_surface < 0 ) // This implies that the user can enter a negative number to deselct any selected cutting surface
            selected_cutting_surface = -1;
          else {
            if(selected_cutting_surface < 1)
              selected_cutting_surface = 1;
            if(selected_cutting_surface > n)
              selected_cutting_surface = n;
          }
        }
      }
      ImGui::SameLine();
      if(ImGui::Button("Erase"))
      {
        if(selected_cutting_surface > 0) {
          viewer.erase_mesh(selected_cutting_surface);
          scene.cuttingSurfaces.erase(scene.cuttingSurfaces.begin() + selected_cutting_surface - 1);
          selected_cutting_surface = -1;
        }
        assert(viewer.data_list.size() == (scene.cuttingSurfaces.size()+1)); 
      }
      ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.20f);
      ImGui::DragInt("# points", &num_random_points_on_cutting_surface, 100, 100, 10000,"%d");
      ImGui::SameLine();
      if(ImGui::Button("Sample Random points"))
      {
        if(selected_cutting_surface > 0) {
          CuttingSurface &surface = scene.cuttingSurfaces.at(selected_cutting_surface-1);

          // Sample random points on the mesh and create indicator function
          sampleRandomPoints(scene.mesh_V, scene.mesh_F, surface.surface_V, surface.surface_F, num_random_points_on_cutting_surface, surface.surfaceRand_V, surface.surfaceRand_V_indicator);
        }
      }
      ImGui::SameLine();
      if(ImGui::Button("Show/Hide##Random points"))
      {
        if((selected_cutting_surface > 0)) {
          size_t npts = viewer.data(selected_cutting_surface).points.rows();
          CuttingSurface &surface = scene.cuttingSurfaces.at(selected_cutting_surface-1);
          if(npts > 0) // Hide
            viewer.data(selected_cutting_surface).clear_points(); 
          else { // Show
            if(surface.surfaceRand_V.rows() > 0) {
              Eigen::MatrixXd C = Eigen::MatrixXd::Zero(surface.surfaceRand_V.rows(), 3);
                C.col(0) = surface.surfaceRand_V_indicator.unaryExpr([](int i){ return (i==MARKER_OUTSIDE)?1.0:0.0;});
                C.col(1) = surface.surfaceRand_V_indicator.unaryExpr([](int i){ return (i==MARKER_BOUNDARY)?1.0:0.0;});
                C.col(2) = surface.surfaceRand_V_indicator.unaryExpr([](int i){ return (i==MARKER_INSIDE)?1.0:0.0;});
              viewer.data(selected_cutting_surface).set_points(surface.surfaceRand_V, C); 
            }
          }
        }
      }

      // Option to sample points around the contour
      ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.20f);
      ImGui::DragFloat("% BBox R", &vicinity_sample_percent_radius, 0.05, 0.1, 5.0, "%.2f");
      ImGui::SameLine();
      if(ImGui::Button("Sample vicinity points"))
      {
        if(selected_cutting_surface > 0) {
          CuttingSurface &surface = scene.cuttingSurfaces.at(selected_cutting_surface-1);
          sampleVicinityPoints(scene.mesh_V, scene.mesh_F, surface.surface_V, surface.surface_F, surface.contour_V, surface.contour_E, surface.contourVicinity_V, surface.contourVicinity_V_indicator);
        }
      }
      ImGui::SameLine();
      if(ImGui::Button("Show/Hide##Vicinity Points"))
      {
        if((selected_cutting_surface > 0)) {
          size_t npts = viewer.data(selected_cutting_surface).points.rows();
          CuttingSurface &surface = scene.cuttingSurfaces.at(selected_cutting_surface-1);
          if(npts > 0) // Hide
            viewer.data(selected_cutting_surface).clear_points(); 
          else { // Show
            if(surface.contourVicinity_V.rows() > 0) {
              Eigen::MatrixXd C = Eigen::MatrixXd::Zero(surface.contourVicinity_V.rows(), 3);
                C.col(0) = surface.contourVicinity_V_indicator.unaryExpr([](int i){ return (i==MARKER_OUTSIDE)?1.0:0.0;});
                C.col(1) = surface.contourVicinity_V_indicator.unaryExpr([](int i){ return (i==MARKER_BOUNDARY)?1.0:0.0;});
                C.col(2) = surface.contourVicinity_V_indicator.unaryExpr([](int i){ return (i==MARKER_INSIDE)?1.0:0.0;});
              viewer.data(selected_cutting_surface).set_points(surface.contourVicinity_V, C); 
            }
          }
        }
      }
      ImGui::DragFloat("% BBox Delta", &vicinity_sample_percent_delta, 0.05, 0.1, 10.0, "%.2f");
      if(ImGui::Button("Sample all"))
      {
        if(scene.cuttingSurfaces.size() > 0) {
          int i = 0;
          std::cout<<"Performing sampling on all cutting surfaces and contours:\n"<<std::flush;
          std::cout<<"\tRandom sampling... "<<std::flush;
          for(auto &surface : scene.cuttingSurfaces) {
            std::cout<<++i<<"... "<<std::flush;
            sampleRandomPoints(scene.mesh_V, scene.mesh_F, surface.surface_V, surface.surface_F, num_random_points_on_cutting_surface, surface.surfaceRand_V, surface.surfaceRand_V_indicator);
          }
          std::cout<<"done.\n";
          i = 0;
          std::cout<<"\tVicinity sampling... "<<std::flush;
          for(auto &surface : scene.cuttingSurfaces) {
            std::cout<<++i<<"... "<<std::flush;
            sampleVicinityPoints(scene.mesh_V, scene.mesh_F, surface.surface_V, surface.surface_F, surface.contour_V, surface.contour_E, surface.contourVicinity_V, surface.contourVicinity_V_indicator);
          }
          std::cout<<"done.\n";
        }
        sampled_all_points = true;
      }
      ImGui::SameLine();
      if(ImGui::Button("Save all"))
      {
        if(!sampled_all_points)
          ImGui::OpenPopup("Save all");
        saveCuttingSurfaces();
      }
      if(ImGui::BeginPopupModal("Save all"))
      {
        ImGui::Text("You have not generated sample points on all cutting surfaces and contours yet.\nPress \"Sample all\" to generate.\n\nNote that this will override any previously generated sample points.");
        if(ImGui::Button("OK"))
        {
          ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
      }
    }
    ImGui::End();
  };

}

// Parameters of the algorithm:
// https://doc.cgal.org/latest/Surface_mesh_skeletonization/classCGAL_1_1Mean__curvature__flow__skeletonization.html#a644afb237cf6941b614fe1a27a5aa916
void computeSkeleton(Eigen::MatrixXd &mV, Eigen::MatrixXi &mF, Eigen::MatrixXd &sV, Eigen::MatrixXi &sE)
{
  typedef CGAL::Simple_cartesian<double>                        Kernel;
  typedef Kernel::Point_3                                       Point;
  typedef CGAL::Polyhedron_3<Kernel>                            Polyhedron;
  typedef boost::graph_traits<Polyhedron>::vertex_descriptor    vertex_descriptor;
  typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron> Skeletonization;
  typedef Skeletonization::Skeleton                             Skeleton;
  typedef Skeleton::vertex_descriptor                           Skeleton_vertex;
  typedef Skeleton::edge_descriptor                             Skeleton_edge;

  Polyhedron mesh;
  igl::copyleft::cgal::mesh_to_polyhedron(mV, mF, mesh);

  Skeleton skeleton;
  Skeletonization mcs(mesh);

  // 1. Contract the mesh by mean curvature flow.
  mcs.contract_geometry();

  // 2. Collapse short edges and split bad triangles.
  mcs.collapse_edges();
  mcs.split_faces();
  
  // 3. Fix degenerate vertices.
  mcs.detect_degeneracies();
  
  // Perform the above three steps in one iteration.
  mcs.contract();
  
  // Iteratively apply step 1 to 3 until convergence.
  mcs.contract_until_convergence();
  
  // Convert the contracted mesh into a curve skeleton and
  // get the correspondent surface points
  mcs.convert_to_skeleton(skeleton);
  std::cout << "Skeleton:\n\tVertices - " << boost::num_vertices(skeleton) << "\n\tEdges -" << boost::num_edges(skeleton) << "\n";
  
  // Convert CGAL::Skeleton to igl data structure
  sV.resize(boost::num_vertices(skeleton), 3); // Skeleton points
  sE.resize(boost::num_edges(skeleton), 2); // Skeleton edges
  size_t idx = 0;
  for(Skeleton_edge e : CGAL::make_range(edges(skeleton)))
  {
      auto src = source(e, skeleton);
      auto dst = target(e, skeleton);

      sE.row(idx++) << src, dst;
      sV.row(src) << skeleton[src].point[0], skeleton[src].point[1], skeleton[src].point[2];
      sV.row(dst) << skeleton[dst].point[0], skeleton[dst].point[1], skeleton[dst].point[2];
  }
}

bool fitCuttingSurface(FittingType ftype)
{
  CuttingSurface surface;
  surface.mesh_pickedPoints = scene.mesh_pickedPoints.topRows(scene.mesh_nPicked);
  surface.skel_pickedPoints = scene.skel_pickedPoints.topRows(scene.skel_nPicked);

  size_t nPoints = surface.mesh_pickedPoints.rows() + surface.skel_pickedPoints.rows();
  size_t minPoints = (ftype==FittingType_Poly33)? 10 : 3;
  if(nPoints < minPoints) {
    std::cout<<"Select at least "<< minPoints <<" points on the mesh and on the skeleton combined to fit a cutting surface.\n";
    return false;
  }

  Eigen::MatrixXd inPoints(nPoints, 3);
  inPoints << surface.mesh_pickedPoints, surface.skel_pickedPoints;

  FitSurface xsection(inPoints, cutting_surface_min_res, scene.mesh_V);
  xsection.fit(surface.surface_V, surface.surface_F, ftype);

  surface.surfaceType = ftype;
  surface.worldToLocal = xsection.getWorldToLocal();
  surface.coefficients = xsection.getCoefficients();

  intersectMeshMesh(scene.mesh_V, scene.mesh_F, surface.surface_V, surface.surface_F, surface.surfaceReg_V, surface.surfaceReg_V_indicator, surface.contours, surface.surfaceCut_V, surface.surfaceCut_F, surface.surfaceCut_F_in, surface.surfaceCut_F_out);

  scene.cuttingSurfaces.push_back(surface); // Add newly created cutting surface to the list

  return true;
}

void saveCuttingSurfaces()
{
  int nsurfaces = scene.cuttingSurfaces.size(); 
  if(nsurfaces == 0) return;
  // Create folder to save
  time_t rawtime;
  struct tm* timeinfo;
  char timebuffer[100];
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(timebuffer, 100, "_sections_%Y%m%d%H%M%S", timeinfo);
  std::filesystem::path savepath = mesh_filename;
  savepath.replace_extension();
  savepath += timebuffer;
  if(!create_directory(savepath)) {
    std::cout<<"Couldn't create output directory: "<< savepath<<". Surfaces not saved!\n"<<std::flush;
    return;
  }
  char filename[512], surfstr[16];
  for(int i=0; i<nsurfaces; i++) {
	  //Create a directory for all the files
		std::filesystem::path surfpath = savepath.string();
    memset(surfstr, 0, sizeof(surfstr));
    snprintf(surfstr, sizeof(surfstr), "/surf%02d", i);
		surfpath += surfstr;
		create_directory(surfpath);

    CuttingSurface &surface = scene.cuttingSurfaces[i];
    // Save seed points
    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/points.pcd", surfpath.c_str());
    std::ofstream ptsfile;
    ptsfile.open(filename);
    ptsfile << "# .PCD v0.7 - Point Cloud Data file format\n";
    ptsfile << "VERSION 0.7\n";
    ptsfile << "FIELDS x y z\n";
    ptsfile << "SIZE 4 4 4\n";
    ptsfile << "TYPE F F F\n";
    ptsfile << "COUNT 1 1 1\n";
    ptsfile << "WIDTH "<< surface.mesh_pickedPoints.rows() + surface.skel_pickedPoints.rows() << std::endl;
    ptsfile << "HEIGHT 1\n";
    ptsfile << "VIEWPOINT 0 0 0 1 0 0 0\n";
    ptsfile << "POINTS " << surface.mesh_pickedPoints.rows() + surface.skel_pickedPoints.rows() << std::endl;
    ptsfile << "DATA ascii\n";
    ptsfile << surface.mesh_pickedPoints;
    ptsfile << surface.skel_pickedPoints;
    ptsfile.close();

    // Save mesh
    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/mesh.obj", surfpath.c_str());
    igl::writeOBJ(filename, surface.surface_V, surface.surface_F);

    // Save surface type, transformation matrix and coefficients
    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/surface.txt", surfpath.c_str());
    std::ofstream surffile;
    surffile.open(filename);
    switch(surface.surfaceType)
    {
    case FittingType_Plane:
      surffile << "Surface type: Plane\n";
      surffile << "z = f(x, y) = 0\n";
      break;
    case FittingType_Poly33:
      surffile << "Surface type: Poly33\n";
      surffile << "z = f(x, y) = a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2 + a6*x^2*y + a7*x*y^2 + a8*x^3 + a9*y^3\n";
      break;
    case FittingType_Biharmonic:
      surffile << "Surface type: Biharmonic\n";
      surffile << "z = f(x, y) = \\sum_{i = 1}^N a_i * d((x, y), (x_i, y_i))^2 * ln(d((x, y), (x_i, y_i)) - 1)\n";
      break;
    }
    surffile << "\nWorld-to-local rigid transformation (PCA) matrix: \n" << surface.worldToLocal<<"\n";
    surffile << "\nCoefficients: \n"<<surface.coefficients;
    surffile.close();

    // Save intersection contours as OBJ
    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/contours.obj", surfpath.c_str());
    std::ofstream contfile;
    contfile.open(filename);
    for(int i=0; i< surface.contour_V.rows(); i++) {
      contfile << "v " << surface.contour_V.row(i) <<"\n";
    }
    for(int i=0; i< surface.contour_E.rows(); i++) {
      contfile << "l " << surface.contour_E.row(i).array() + 1 <<"\n";
    }
    contfile.close();

    // Save contours as loops in a text file
    // Format: 
    // Line 1 - total number of contours
    // Line 2 - # nodes in contour 1
    // Line 3 - nodes in contour 1 (indexing vertices in reg_indicator.txt, 0-based)
    // subsequent lines repeat lines 2 and 3 as may times as there are contours.
    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/contours.txt", surfpath.c_str());
    std::ofstream contfile2;
    contfile2.open(filename);
    contfile2 <<  surface.contours.size() << "\n";
    for(auto loop : surface.contours) {
      contfile2 << loop.size() << "\n";
      for(int i : loop)
        contfile2 << i << " ";
      contfile2 << "\n";
    }
    contfile2.close();

    // Save indicator function of regular sample points of the surface mesh + cut points
    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/reg_indicator.txt", surfpath.c_str());
    std::ofstream regindfile;
    regindfile.open(filename);

    // Save only points inside and outside the mesh (i.e, exclude points on the boundary)
    std::vector<int> indices_keep;
    size_t nrows = surface.surfaceReg_V_indicator.rows();
    for(int i=0; i<nrows; i++)
        if(surface.surfaceReg_V_indicator(i) != MARKER_BOUNDARY) indices_keep.push_back(i);
    for(int i=0; i< indices_keep.size(); i++) {
      regindfile << surface.surfaceReg_V.row(indices_keep[i]) << " " << surface.surfaceReg_V_indicator(indices_keep[i]) << "\n";
    }
    regindfile.close();

    // Save indicator function of random sample points of the surface mesh + cut points
    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/rnd_indicator.txt", surfpath.c_str());
    std::ofstream rndindfile;
    rndindfile.open(filename);
    for(int i=0; i< surface.surfaceRand_V.rows(); i++) {
      rndindfile << surface.surfaceRand_V.row(i) << " " << surface.surfaceRand_V_indicator(i) << "\n";
    }
    rndindfile.close();

    // Save indicator function of vicinity sample points of the contours
    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/vic_indicator.txt", surfpath.c_str());
    std::ofstream vicindfile;
    vicindfile.open(filename);
    for(int i=0; i< surface.contourVicinity_V.rows(); i++) {
      vicindfile << surface.contourVicinity_V.row(i) << " " << surface.contourVicinity_V_indicator(i) << "\n";
    }
    vicindfile.close();
    
    // Save cross-section mesh: all faces, inside faces, outside faces
    Eigen::MatrixXi xF;
    Eigen::MatrixXd xV;
    Eigen::VectorXi xI;

    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/meshx.obj", surfpath.c_str());
    igl::remove_unreferenced(surface.surfaceCut_V, surface.surfaceCut_F, xV, xF, xI);
    igl::writeOBJ(filename, xV, xF);

    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/meshx_in.obj", surfpath.c_str());
    xF.resize(0, 3); xV.resize(0, 3);
    igl::remove_unreferenced(surface.surfaceCut_V, surface.surfaceCut_F_in, xV, xF, xI);
    igl::writeOBJ(filename, xV, xF);

    memset(filename, 0, sizeof(filename));
    snprintf(filename, sizeof(filename), "%s/meshx_out.obj", surfpath.c_str());
    xF.resize(0, 3); xV.resize(0, 3);
    igl::remove_unreferenced(surface.surfaceCut_V, surface.surfaceCut_F_out, xV, xF, xI);
    igl::writeOBJ(filename, xV, xF);

  }
  std::cout<<"Saved "<<nsurfaces<<" cross-sections in "<<savepath.c_str()<<"\n"<<std::flush;
}

void intersectMeshMesh(Eigen::MatrixXd &V_solid, Eigen::MatrixXi &F_solid, Eigen::MatrixXd &V_surf, Eigen::MatrixXi &F_surf, Eigen::MatrixXd &V_surfReg, Eigen::VectorXi &V_surfRegIndicator, std::vector<std::vector<int> > &contours, Eigen::MatrixXd &V_Cut, Eigen::MatrixXi &F_Cut, Eigen::MatrixXi &F_Cut_in, Eigen::MatrixXi &F_Cut_out)
{
  Eigen::VectorXi F_out_pred; // Boolean for inside/outside
  Eigen::VectorXi F_idx;
  igl::copyleft::cgal::trim_with_solid(V_surf, F_surf, V_solid, F_solid, V_Cut, F_Cut, F_out_pred, F_idx);

  //Extract boundary loops
  F_Cut_in.resize(F_Cut.rows() - F_out_pred.sum(), 3);
  F_Cut_out.resize(F_out_pred.sum(), 3);
  for(size_t i = 0, j = 0, k = 0; i< F_out_pred.rows(); i++) // Keep faces that are inside/outside
    if(!F_out_pred(i)) F_Cut_in.row(j++) = F_Cut.row(i);
    else F_Cut_out.row(k++) = F_Cut.row(i);
  
  igl::boundary_loop(F_Cut_in, contours);
  std::cout<<"Found "<< contours.size() <<" loops in intersection.\n"<<std::flush;

  // Create indicator function for all vertices in the original vertices and added vertices of the cut: V_Cut
  Eigen::VectorXi V_surfRegIndicatorAll(V_Cut.rows());
  igl::slice_into(F_out_pred.unaryExpr([](int i){return 1-2*i;}), F_Cut.col(0), V_surfRegIndicatorAll);
  igl::slice_into(F_out_pred.unaryExpr([](int i){return 1-2*i;}), F_Cut.col(1), V_surfRegIndicatorAll);
  igl::slice_into(F_out_pred.unaryExpr([](int i){return 1-2*i;}), F_Cut.col(2), V_surfRegIndicatorAll);
  //Fix zero level points
  for(auto curve: contours) {
    igl::slice_into(Eigen::VectorXi::Zero(curve.size()), Eigen::Map<Eigen::VectorXi>(curve.data(), curve.size()), V_surfRegIndicatorAll);
  }
  // Convert -1, 0, 1 to defined marker types
  V_surfRegIndicatorAll = V_surfRegIndicatorAll.unaryExpr([&](int i)->int{
    return (i==0)?MARKER_BOUNDARY:(i==-1)?MARKER_OUTSIDE:(i==1)?MARKER_INSIDE:MARKER_UNUSED;});

  // Retain points from V_cut those are only within the solid mesh BBOX
  std::vector<int> keepIndices_vec;
  Eigen::RowVectorXd bbox_min = V_solid.colwise().minCoeff();
  Eigen::RowVectorXd bbox_max = V_solid.colwise().maxCoeff();
  Eigen::RowVectorXd pt;
  for (int i = 0; i < V_Cut.rows(); i++){
    if( V_Cut(i, 0) < bbox_min(0) || V_Cut(i, 0) > bbox_max(0) ||
        V_Cut(i, 1) < bbox_min(1) || V_Cut(i, 1) > bbox_max(1) ||
        V_Cut(i, 2) < bbox_min(2) || V_Cut(i, 2) > bbox_max(2)) // Reject point outside mesh BBOX
        continue;
    keepIndices_vec.push_back(i);
  }
  Eigen::VectorXi keepIndices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(keepIndices_vec.data(), keepIndices_vec.size());
  V_surfReg.resize(keepIndices_vec.size(), 3);
  V_surfRegIndicator.resize(keepIndices_vec.size());
  igl::slice(V_Cut, keepIndices, 1, V_surfReg);
  igl::slice(V_surfRegIndicatorAll, keepIndices, 1, V_surfRegIndicator);
}

void contourChainsToEdges(const std::vector<std::vector<int> > &chains, Eigen::MatrixXi &E)
{
  size_t E_sz = 0;
  for(auto curve : chains) E_sz += curve.size();
  if(E_sz == 0) return;

  E.resize(E_sz, 2);
  size_t i=0;
  for(auto curve : chains) {
    E.block(i, 0, curve.size(), 1) = Eigen::Map<Eigen::VectorXi>(curve.data(), curve.size());
    E.block(i, 1, curve.size()-1, 1) = Eigen::Map<Eigen::VectorXi>(curve.data()+1, curve.size()-1);
    E(i+curve.size()-1,1) = curve[0];
    i+=curve.size();
  }
}

void sampleRandomPoints(const Eigen::MatrixXd &meshV, const Eigen::MatrixXi &meshF, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, size_t nsamples, Eigen::MatrixXd &samples, Eigen::VectorXi &samplesIndicator)
{
  Eigen::VectorXi FI;
  Eigen::MatrixXd B;
  igl::random_points_on_mesh(nsamples, V, F, B, FI);
 
  Eigen::MatrixXd samplePoints(B.rows(), 3);
  Eigen::RowVectorXd bbox_min = meshV.colwise().minCoeff();
  Eigen::RowVectorXd bbox_max = meshV.colwise().maxCoeff();
  Eigen::RowVectorXd pt;
  size_t npts = 0;
  for (int i = 0; i < B.rows(); i++){
    pt = B(i, 0)*V.row(F(FI(i), 0)) +
         B(i, 1)*V.row(F(FI(i), 1)) +
         B(i, 2)*V.row(F(FI(i), 2));
    if( pt(0) < bbox_min(0) || pt(0) > bbox_max(0) ||
        pt(1) < bbox_min(1) || pt(1) > bbox_max(1) ||
        pt(2) < bbox_min(2) || pt(2) > bbox_max(2)) // Reject point outside mesh BBOX
        continue;
    samplePoints.row(npts++) = pt;
  }
  samples.resize(npts, 3);
  samples = samplePoints.block(0, 0, npts, 3);// Copy only valid points to samples array

  Eigen::VectorXi inside;

  igl::copyleft::cgal::points_inside_component(meshV, meshF, samples, inside);
  // Create indicator function
  samplesIndicator.resize(samples.rows());
  samplesIndicator = 2*inside.array() - 1;
  // Convert -1, 0, 1 to defined marker types
  samplesIndicator  = samplesIndicator.unaryExpr([&](int i)->int{
          return (i==0)?MARKER_BOUNDARY:(i==-1)?MARKER_OUTSIDE:(i==1)?MARKER_INSIDE:MARKER_UNUSED;});
}

void sampleVicinityPoints(const Eigen::MatrixXd &meshV, const Eigen::MatrixXi &meshF, const Eigen::MatrixXd &surfV, const Eigen::MatrixXi &surfF, const Eigen::MatrixXd &contV, const Eigen::MatrixXi &contE, Eigen::MatrixXd &vicinityV, Eigen::MatrixXi &vicinityIndicator)
{
  float bbox_diag = igl::bounding_box_diagonal(meshV);
  float rad = (vicinity_sample_percent_radius/100.0)*bbox_diag;
  float delta = (vicinity_sample_percent_delta/100.0)*bbox_diag;
  int numfacets = 6;

  // Subsample contour edges (optional?)

  // Inflate/offset the contour
  Eigen::MatrixXd OV;
  Eigen::MatrixXi OF, OJ;
  igl::copyleft::cgal::wire_mesh(contV, contE, rad*2, numfacets, true, OV, OF, OJ); 

  // Intersect Inflated mesh with cutting surface and cleanup
  Eigen::VectorXi OF_out_pred; // Boolean for inside/outside
  Eigen::VectorXi OF_idx;
  Eigen::MatrixXd OV_cut;
  Eigen::MatrixXi OF_cut;
  igl::copyleft::cgal::trim_with_solid(surfV, surfF, OV, OF, OV_cut, OF_cut, OF_out_pred, OF_idx);
  Eigen::MatrixXi OF_in;
  OF_in.resize(OF_cut.rows() - OF_out_pred.sum(), 3);
  for(size_t i = 0, j = 0; i< OF_out_pred.rows(); i++) // Keep faces that are inside/outside
    if(!OF_out_pred(i)) OF_in.row(j++) = OF_cut.row(i);
  
  std::vector<std::vector<int> > vicinity_contours;
  igl::boundary_loop(OF_in, vicinity_contours);
  if(vicinity_contours.size() % 2 != 0) // This must be even
    std::cout<<"WARNING: Vicinity contours not even in number ("<< vicinity_contours.size() <<").\n"<<std::flush;

  // Resample curves
  std::vector<std::vector<int> > resampled_contours;
  std::vector<Eigen::MatrixXd> resampled_contoursV;
  std::cout<<"Resampled contours (points): ";
  for(auto & contour : vicinity_contours)
  {
    std::vector<int> recontour;
    Eigen::MatrixXd recontourV;
    resampleContour(delta, contour, OV_cut, recontour, recontourV);
    resampled_contours.push_back(recontour);
    resampled_contoursV.push_back(recontourV);
    std::cout<<"("<<contour.size()<<"->"<<recontour.size()<<"), ";
  }
  std::cout<<"\n"<<std::flush;

  // Resize output arrays
  size_t total_sz = 0;
  for(int i=0; i<resampled_contours.size(); i++) total_sz += resampled_contours[i].size();
  vicinityV.resize(total_sz, 3);
  vicinityIndicator.resize(total_sz, 1);

  // Check inside outside w.r.t. mesh. classify contours as inside/outside.
  size_t idx = 0;
  typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
  for(int i=0; i<resampled_contours.size(); i++) {
    ArrayXb inside;
    const Eigen::MatrixXd &points = resampled_contoursV[i];
    igl::copyleft::cgal::points_inside_component(meshV, meshF, points, inside);

    vicinityV.block(idx, 0, points.rows(), 3) = points;
    vicinityIndicator.block(idx, 0, inside.rows(), 1) = inside.matrix().cast<int>();
    idx += points.rows();
  }
  vicinityIndicator = vicinityIndicator.unaryExpr([&](int i)->int{ return (i==0)?MARKER_OUTSIDE:MARKER_INSIDE;});
}

void resampleContour(double delta, std::vector<int> contour, const Eigen::MatrixXd &points, std::vector<int> &recontour, Eigen::MatrixXd &repoints)
{
  std::vector<double> dists;
  std::vector<double> cumdists;
  std::vector<double> repoints_vec;
  if(contour.front() != contour.back())
    contour.push_back(contour[0]);
  for(int j=0; j<contour.size()-1; j++)
    dists.push_back((points.row(contour[j+1]) - points.row(contour[j])).norm());
  cumdists.push_back(dists[0]);
  for(int j=1; j<contour.size()-1; j++)
    cumdists.push_back(cumdists[j-1] + dists[j]);

  // Starting with the first point, resample the contour
  Eigen::RowVector3d p = points.row(contour[0]);
  const double *ptr = p.data();
  repoints_vec.insert(repoints_vec.end(), ptr, ptr+3);
  recontour.push_back(0);
  double kdelta = delta; //kdelta is the distance traversed along the curve
  double fraction;

  while(true)
  {
    auto lower_iter = std::lower_bound(cumdists.begin(), cumdists.end(), kdelta); // kdelta <= cumdists[lower]
    if(lower_iter == cumdists.end()) break;// break from loop and examine the last point inserted in recontour w.r.t. last point on contour
    int idx = std::distance(cumdists.begin(), lower_iter);

    // Between [idx, idx+1] calculate a point 
    const Eigen::RowVector3d p0 = points.row(contour[idx]);
    const Eigen::RowVector3d p1 = points.row(contour[idx+1]);
    fraction = (*lower_iter - kdelta) / (p0-p1).norm(); // This is the fraction of distance p is away from p1
    p = p1 + fraction*(p0-p1);
    ptr = p.data();
    repoints_vec.insert(repoints_vec.end(), ptr, ptr+3);
    recontour.push_back(repoints_vec.size()/3 - 1);
    kdelta += delta;
  }
  if( (points.row(contour[0]) - p).norm() < delta/2) // the remove the last added points - Just a heuristic to keep inter-point distance >=delta
  {
    recontour.pop_back();
    for(int i=0; i<3; i++) repoints_vec.pop_back();
  }

  // Convert vector of points to Eigen matrix
  repoints = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > (repoints_vec.data(), repoints_vec.size()/3, 3);
}
