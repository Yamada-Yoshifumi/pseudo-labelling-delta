import sys
from SyMBac.simulation import Simulation
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
import tifffile
from SyMBac.misc import get_sample_images

def scene_generation(length, length_var, width, width_var, sim_length, sample_size, initialised, save_dir):

    real_image = tifffile.imread("/home/ameyasu/cuda_ws/src/pseudo_labelling_real/short_trench_sample.tif")

    my_simulation = Simulation(
        trench_length=12,
        trench_width=1.3,
        #cell_max_length=1.4, #6, long cells # 1.65 short cells
        #cell_width= 1, #1 long cells # 0.95 short cells
        #sim_length = 100,
        cell_max_length=length,
        cell_width=width,
        sim_length=sim_length,
        pix_mic_conv = 0.065,
        gravity=0,
        phys_iters=15,
        #max_length_var = 0.,
        #width_var = 0.,
        max_length_var=length_var,
        width_var=width_var,
        lysis_p = 0.,
        save_dir="/home/ameyasu/cuda_ws/src/pseudo_labelling_real",
        resize_amount = 3
    )

    my_simulation.run_simulation(show_window=False)

    my_simulation.draw_simulation_OPL(do_transformation=True, label_masks=True)

    my_kernel = PSF_generator(
        radius = 50,
        wavelength = 0.75,
        NA = 1.2,
        n = 1.3,
        resize_amount = 3,
        pix_mic_conv = 0.065,
        apo_sigma = 20,
        mode="phase contrast",
        condenser = "Ph3")
    my_kernel.calculate_PSF()
    #my_kernel.plot_PSF()

    my_camera = Camera(baseline=100, sensitivity=2.9, dark_noise=8)
    #my_camera.render_dark_image(size=(300,300))

    my_renderer = Renderer(simulation = my_simulation, PSF = my_kernel, real_image = real_image, camera = my_camera)

    if not initialised:
        my_renderer.select_intensity_napari()

    my_renderer.optimise_synth_image(manual_update=False, api_access=True)

    my_renderer.generate_training_data(sample_amount=0.1, randomise_hist_match=True, randomise_noise_match=True, 
                                    burn_in=40, n_samples = sample_size, save_dir=save_dir, in_series=False)


if __name__ == "__main__":
    scene_generation(length = 3, length_var = 0.5, width = 0.95, width_var = 0.2, sim_length = 600, sample_size=500, initialised = True, save_dir='/tmp/test/')
