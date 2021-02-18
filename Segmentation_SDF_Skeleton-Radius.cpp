#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/mesh_segmentation.h>
#include <CGAL/property_map.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <iostream>
#include <fstream>
//typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Simple_cartesian<double>                               Kernel;
//typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> Polyhedron;
typedef boost::graph_traits<Polyhedron>::face_descriptor face_descriptor;
typedef Kernel::Point_3                                              Point;
typedef boost::graph_traits<Polyhedron>::vertex_descriptor           vertex_descriptor;
typedef boost::graph_traits<Polyhedron>::halfedge_descriptor         halfedge_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron>        Skeletonization;
typedef Skeletonization::Skeleton                                    Skeleton;
typedef Skeleton::vertex_descriptor                                  Skeleton_vertex;

// Property map associating a facet with an integer as id to an
// element in a vector stored internally
template<class ValueType>
struct Facet_with_id_pmap
    : public boost::put_get_helper<ValueType&,
    Facet_with_id_pmap<ValueType> >
{
    typedef face_descriptor key_type;
    typedef ValueType value_type;
    typedef value_type& reference;
    typedef boost::lvalue_property_map_tag category;
    Facet_with_id_pmap(
        std::vector<ValueType>& internal_vector
    ) : internal_vector(internal_vector) { }
    reference operator[](key_type key) const
    {
        return internal_vector[key->id()];
    }
private:
    std::vector<ValueType>& internal_vector;
};
int main(int argc, char** argv)
{
    // create and read Polyhedron
    std::ifstream input((argc > 1) ? argv[1] : "data/Kasthuri__0025_Spines.D4_Spines.D4_Spine_8_A.off"); // 31_R Kasthuri__0019_Spines.D4_B1_Spine_2_-.off
    Polyhedron mesh;
    input >> mesh;
    if (!CGAL::is_triangle_mesh(mesh))
    {
        std::cout << "Input geometry is not triangulated." << std::endl;
        return EXIT_FAILURE;
    }
    // create a property-map for SDF values
    typedef std::map<face_descriptor, double> Facet_double_map;
    Facet_double_map internal_sdf_map;
    boost::associative_property_map<Facet_double_map> sdf_property_map(internal_sdf_map);
    // compute SDF values using default parameters for number of rays, and cone angle
    CGAL::sdf_values(mesh, sdf_property_map);
    //
    // start Netanel
    //
    // extract the skeleton
    Skeleton skeleton;
    CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);
    // init the polyhedron simplex indices
    CGAL::set_halfedgeds_items_id(mesh);
    //for each input vertex compute its distance to the skeleton
    std::vector<double> distances(num_vertices(mesh));
    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
    {
        const Point& skel_pt = skeleton[v].point;
        for (vertex_descriptor mesh_v : skeleton[v].vertices)
        {
            const Point& mesh_pt = mesh_v->point();
            distances[mesh_v->id()] = std::sqrt(CGAL::squared_distance(skel_pt, mesh_pt));
        }
    }
    // create a property-map for SDF values
    //typedef std::map<face_descriptor, double> Facet_double_map;
    //Facet_double_map internal_sdf_map;
    //boost::associative_property_map<Facet_double_map> sdf_property_map2(internal_sdf_map);
    //
    // create a property-map for sdf values
    std::vector<double> sdf_values(num_faces(mesh));
    Facet_with_id_pmap<double> sdf_property_map2(sdf_values);
    // compute sdf values with skeleton
    for (face_descriptor f : faces(mesh))
    {
        double dist = 0;
        for (halfedge_descriptor hd : halfedges_around_face(halfedge(f, mesh), mesh))
            dist += distances[target(hd, mesh)->id()];
        sdf_property_map2[f] = dist / 3.;
    }
    // post-process the sdf values
    //CGAL::sdf_values_postprocessing(mesh, sdf_property_map2);
    
    double currentMax = 0.0;
    double currentMin = 100.0;
    for (face_descriptor f : faces(mesh))
    {
        if (sdf_property_map2[f] > currentMax) {
            currentMax = sdf_property_map2[f];
        }
        if (sdf_property_map2[f] < currentMin) {
            currentMin = sdf_property_map2[f];
        }
    }
    //std::cout << currentMax << " ";
    //std::cout << currentMin << " ";
    //
    for (face_descriptor f : faces(mesh))
    {
        //sdf_property_map[f] = (sdf_property_map[f] + sdf_property_map2[f])/2.0;
        sdf_property_map[f] = (sdf_property_map[f] + ((sdf_property_map2[f] - currentMin) / (currentMax - currentMin))) / 2.0;
    }
    //
    // end Netanel
    //
    // create a property-map for segment-ids
    typedef std::map<face_descriptor, std::size_t> Facet_int_map;
    Facet_int_map internal_segment_map;
    boost::associative_property_map<Facet_int_map> segment_property_map(internal_segment_map);
    // segment the mesh using default parameters for number of levels, and smoothing lambda
    // Any other scalar values can be used instead of using SDF values computed using the CGAL function
    const std::size_t number_of_clusters = 2;       // use 4 clusters in soft clustering
    const double smoothing_lambda = 0.1;  // importance of surface features, suggested to be in-between [0,1]
    std::size_t number_of_segments = CGAL::segmentation_from_sdf_values(mesh, sdf_property_map, segment_property_map, number_of_clusters, smoothing_lambda);
    //std::cout << "Number of segments: " << number_of_segments << std::endl;
    // print segment-ids
    for (face_descriptor f : faces(mesh)) {
        // ids are between [0, number_of_segments -1]
        std::cout << segment_property_map[f] << " ";
    }
    std::cout << std::endl;
    // Note that we can use the same SDF values (sdf_property_map) over and over again for segmentation.
    // This feature is relevant for segmenting the mesh several times with different parameters.
    //CGAL::segmentation_from_sdf_values(mesh, sdf_property_map, segment_property_map, number_of_clusters, smoothing_lambda);
    return EXIT_SUCCESS;
}



