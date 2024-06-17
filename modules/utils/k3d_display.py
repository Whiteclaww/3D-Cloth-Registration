import k3d

from k3d.colormaps import matplotlib_color_maps

def representation(garment, new_garment, smpl, distances, vectors, norms, centers, why):

    kdsmpl = k3d.points(
        positions = smpl,
        name = "smpl",
        point_size = 0.01,
        opacity = 1,
        colomap = matplotlib_color_maps.Jet,
        attribute = distances,
        color_range = [0, 0.005],
        shader = "mesh",
        compression_level = 9
    )

    kdgarment = k3d.points(
        positions = garment.vertices,
        color = 98888,
        name = "garment",
        point_size = 0.01,
        opacity = 1,
        colomap = matplotlib_color_maps.Jet,
        shader = "mesh",
        compression_level = 9
    )

    kdnewgarment = k3d.points(
        positions = new_garment,
        name = "new garment",
        point_size = 0.01,
        opacity = 1,
        colomap = matplotlib_color_maps.Jet,
        shader = "mesh",
        compression_level = 9
    )

    plt_vectors = k3d.vectors(
        why,
        vectors,
        head_size=0.01,
        line_width=0.001
    )
    
    '''plt_norms = k3d.vectors(
        centers,
        norms,
        line_width=0.02)'''
    
    # display vectors colorgradient if vector size >>>
    # vectors origins
    # vectors are M to projected point, norms are face norms to check for inconsistencies

    plot = k3d.plot(grid_visible = True, snapshot_type = "full", menu_visibility = True)
    plot += kdsmpl
    plot += kdgarment
    plot += kdnewgarment
    plot += plt_vectors
    #plot += plt_norms

    return plot

# save 
def save(path, plot):
    with open(path + ".html", "w") as f:
        f.write(plot.get_snapshot())