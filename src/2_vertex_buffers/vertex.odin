package vertex_buffers

import "base:runtime"
import "core:fmt"
import "core:strings"
import "core:math"
import "core:mem"
import glm "core:math/linalg/glsl"
import "vendor:glfw"
import vk "vendor:vulkan"

U32_MAX: u32 : 4294967295
U64_MAX: u64 : 18446744073709551615
MAX_FRAME_IN_FLIGHT :: 2

WindowHandle :: glfw.WindowHandle

Window :: struct {
	handle: WindowHandle,
	name: cstring,
	width, height: i32,
}

Renderer :: struct {
	instance: vk.Instance,
	debug_messenger: vk.DebugUtilsMessengerEXT,
	physical_device: vk.PhysicalDevice,
	device: vk.Device,
	graphics_queue: vk.Queue,
	present_queue: vk.Queue,
	surface: vk.SurfaceKHR,
	swap_chain: vk.SwapchainKHR,
	swap_chain_images: [dynamic]vk.Image,
	swap_chain_image_format: vk.Format,
	swap_chain_extent: vk.Extent2D,
	swap_chain_image_views: [dynamic]vk.ImageView,
	render_pass: vk.RenderPass,
	pipeline_layout: vk.PipelineLayout,
	graphics_pipeline: vk.Pipeline,
	swap_chainbuffers: [dynamic]vk.Framebuffer,
	command_pool: vk.CommandPool,
	command_buffers: [dynamic]vk.CommandBuffer,
	image_available_semaphores: [dynamic]vk.Semaphore,
	render_finished_semaphores: [dynamic]vk.Semaphore,
	in_flight_fences: [dynamic]vk.Fence,
	vertex_buffer: vk.Buffer,
	vertex_buffer_memory: vk.DeviceMemory,
	index_buffer: vk.Buffer,
	index_buffer_memory: vk.DeviceMemory,
}

Queue_Family_Indices :: struct {
	graphics_family: Maybe(u32),
	present_family: Maybe(u32),
}

Swap_Chain_Support_Details :: struct {
	capabilities: vk.SurfaceCapabilitiesKHR, 
	formats: []vk.SurfaceFormatKHR,
	present_modes: []vk.PresentModeKHR,
}

Vertex :: struct {
	pos: glm.vec2,
	color: glm.vec3,
}

validation_layers := []cstring{"VK_LAYER_KHRONOS_validation"}
device_extensions := []cstring{vk.KHR_SWAPCHAIN_EXTENSION_NAME}

window_info: Window
renderer: Renderer

current_frame: u32
frame_buffer_resized: bool

vertices := []Vertex {
    {{-0.5, -0.5}, {1.0, 0.0, 0.0}},
    {{ 0.5, -0.5}, {0.0, 1.0, 0.0}},
    {{ 0.5,  0.5}, {0.0, 0.0, 1.0}},
    {{-0.5,  0.5}, {1.0, 1.0, 1.0}},
}


/*
    v0         v1
    *--------- *
    |          |
    |          |
    *--------- *
    v2         v3
*/

indices := []u16 {
	// Upper Triangle
	0, 1, 2, 
	// Lower Triangle
	2, 3, 0,
}

vertex_shader_code :: #load("shader.vert.spv")
fragment_shader_code :: #load("shader.frag.spv")

vk_check_result :: proc(result: vk.Result, loc := #caller_location) {
	assert(result == .SUCCESS, 
		   fmt.tprintf("Procedure: %s - Line: %d - Vulkan Error: %v", 
			           loc.procedure, loc.line, result))
}

vk_check_swap :: proc(result: vk.Result, loc := #caller_location) {
	if result == .ERROR_OUT_OF_DATE_KHR { 
		recreate_swap_chain()
		return 
	}
	assert(result == .SUCCESS || result == .SUBOPTIMAL_KHR, 
		   fmt.tprintf("Procedure: %s - Line: %d - Vulkan Error: %v", 
			           loc.procedure, loc.line, result))
}

vk_check_queue :: proc(result: vk.Result, loc := #caller_location) {
	if result == .ERROR_OUT_OF_DATE_KHR || result == .SUBOPTIMAL_KHR {
		frame_buffer_resized = false
		recreate_swap_chain()
		return
	}
	assert(result == .SUCCESS, 
		   fmt.tprintf("Procedure: %s - Line: %d - Vulkan Error: %v", 
					   loc.procedure, loc.line, result))
}

create_instance :: proc() {
	vk.load_proc_addresses(rawptr(glfw.GetInstanceProcAddress))

	app_info := vk.ApplicationInfo {
		sType = .APPLICATION_INFO,
		pApplicationName = "Hello Triangle",
		applicationVersion = vk.MAKE_VERSION(1, 0, 0),
		pEngineName = "No engine",
		engineVersion = vk.MAKE_VERSION(1, 0, 0),
		apiVersion = vk.API_VERSION_1_2,
	}

	create_info := vk.InstanceCreateInfo {
		sType = .INSTANCE_CREATE_INFO,
		pNext = nil,
		pApplicationInfo = &app_info,
	}

	debug_create_info: vk.DebugUtilsMessengerCreateInfoEXT

	when ODIN_DEBUG {
		result := check_validation_layer_support()
		assert(result, "Failed to find all validation layers")
		create_info.enabledLayerCount = u32(len(validation_layers))
		create_info.ppEnabledLayerNames = raw_data(validation_layers)

		debug_create_info = vk.DebugUtilsMessengerCreateInfoEXT {
			sType = .DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
			messageSeverity = {.VERBOSE, .WARNING, .ERROR},
			messageType = {.GENERAL, .VALIDATION, .PERFORMANCE},
			pfnUserCallback = debug_callback,
		}
		populate_debug_messenger(&debug_create_info)
		create_info.pNext = (^vk.DebugUtilsMessengerCreateInfoEXT)(&debug_create_info)
	} 

	required_extensions := get_required_extensions()

	create_info.enabledExtensionCount = u32(len(required_extensions))
	create_info.ppEnabledExtensionNames = raw_data(required_extensions)

	when ODIN_OS == .Darwin {
		create_info.flags = {.ENUMERATE_PORTABILITY_KHR}
	}

	vk_check_result(vk.CreateInstance(&create_info, nil, &renderer.instance))

	vk.load_proc_addresses_instance(renderer.instance)
}

check_validation_layer_support :: proc() -> bool {
	layer_found := false

	layer_count: u32
	vk.EnumerateInstanceLayerProperties(&layer_count, nil)

	available_layers := make([]vk.LayerProperties, layer_count)
	vk.EnumerateInstanceLayerProperties(&layer_count, raw_data(available_layers))

	outer: for layer_name in validation_layers {
		for layer_properties in available_layers {
			bytes := layer_properties.layerName
        	builder := strings.clone_from_bytes(bytes[:])
        	layer_name_cstring := strings.clone_to_cstring(builder)
			if layer_name == layer_name_cstring {
				layer_found = true
				break outer
			}
		}
	}

	return layer_found
}

get_required_extensions :: proc() -> []cstring {
	required_extensions: [dynamic]cstring

	extensions_names := glfw.GetRequiredInstanceExtensions()
	extensions_count := u32(len(extensions_names))

	for e in extensions_names {
		append(&required_extensions, e)
	}

	when ODIN_OS == .Darwin {
		append(&required_extensions, vk.KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
	}

	when ODIN_DEBUG {
		append(&required_extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
	}

	return required_extensions[:]
}

setup_debug_messenger :: proc() {
	create_info: vk.DebugUtilsMessengerCreateInfoEXT
	populate_debug_messenger(&create_info)

	vk_check_result(vk.CreateDebugUtilsMessengerEXT(renderer.instance, &create_info, 
		                                            nil, &renderer.debug_messenger))
}

populate_debug_messenger :: proc(create_info: ^vk.DebugUtilsMessengerCreateInfoEXT) {
	create_info.sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT
	create_info.messageSeverity = {.VERBOSE, .WARNING, .ERROR}
	create_info.messageType = {.GENERAL, .VALIDATION, .PERFORMANCE}
	create_info.pfnUserCallback = debug_callback
}

debug_callback :: proc "system" (
    messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
    messageTypes: vk.DebugUtilsMessageTypeFlagsEXT,
    pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
    pUserData: rawptr,
) -> b32 {
	context = runtime.default_context()

	result: b32 = false
    fmt.eprintln("Vulkan Debug Message:", pCallbackData.pMessage)

    if .ERROR in messageSeverity {
		result = true
    }

    return result
}

create_surface :: proc() {
	vk_check_result(glfw.CreateWindowSurface(renderer.instance, window_info.handle, 
		                                     nil, &renderer.surface))
}

pick_physical_device :: proc() {
	device_count: u32
	vk_check_result(vk.EnumeratePhysicalDevices(renderer.instance, &device_count, nil))
	assert(device_count > 0, "Could not find GPU with Vulkan support!")
	devices := make([]vk.PhysicalDevice, device_count)
	vk_check_result(vk.EnumeratePhysicalDevices(renderer.instance, &device_count, raw_data(devices)))

	for device in devices {
        if is_device_suitable(device) {
			renderer.physical_device = device
			break
		}
	}

	assert(renderer.physical_device != nil, "Failed to find a suitable GPU!")
}

is_device_suitable :: proc(physical_device: vk.PhysicalDevice) -> bool {
	indices := find_queue_families(physical_device)
	has_required_queues := indices.graphics_family != nil && indices.present_family != nil
	has_required_extensions := check_device_extension_support(physical_device)
	
	swap_chain_adequate := false
	if (has_required_extensions) {
		swap_chain_support := query_swap_chain_support(physical_device)
		swap_chain_adequate = swap_chain_support.formats != nil && swap_chain_support.present_modes != nil
	}

	return has_required_queues && has_required_extensions && swap_chain_adequate
}

find_queue_families :: proc(physical_device: vk.PhysicalDevice) -> Queue_Family_Indices {
	indices: Queue_Family_Indices
	queue_family_count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nil)

	queue_families := make([]vk.QueueFamilyProperties, queue_family_count)
	vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, 
		                                      raw_data(queue_families))

	for queue_family, i in queue_families {
		if .GRAPHICS in queue_family.queueFlags do indices.graphics_family = u32(i)

		present_support: b32
		vk_check_result(vk.GetPhysicalDeviceSurfaceSupportKHR(physical_device, u32(i), 
															  renderer.surface, &present_support))
		if present_support {
			indices.present_family = u32(i)
		}

		if indices.graphics_family != nil do break
	}

	return indices
}

check_device_extension_support :: proc(physical_device: vk.PhysicalDevice) -> bool {
	device_found := false

	extension_count: u32
	vk_check_result(vk.EnumerateDeviceExtensionProperties(physical_device, nil, 
		                                                  &extension_count, nil))

	available_extensions := make([]vk.ExtensionProperties, extension_count)
	vk_check_result(vk.EnumerateDeviceExtensionProperties(physical_device, nil, 
		                                                  &extension_count, raw_data(available_extensions)))

	outer: for extension_name in device_extensions {
		for extension in available_extensions {
			bytes := extension.extensionName
        	builder := strings.clone_from_bytes(bytes[:])
        	extension_name_cstring := strings.clone_to_cstring(builder)
			if extension_name == extension_name_cstring {
				device_found = true
				break outer
			}
		}
	}

	return device_found
}

create_logical_device :: proc() {
	indices := find_queue_families(renderer.physical_device)
	assert(indices.graphics_family != nil)
	assert(indices.present_family != nil)

	unique_queue_families: [dynamic]u32
	append(&unique_queue_families, indices.graphics_family.?)

	if indices.graphics_family.? != indices.present_family.? {
		append(&unique_queue_families, indices.present_family.?)
	}

	queue_create_infos := make([]vk.DeviceQueueCreateInfo, len(unique_queue_families))

	queue_priority: f32 = 1.0

	for queue_family, i in unique_queue_families {
		queue_create_infos[i] = vk.DeviceQueueCreateInfo {
			sType = .DEVICE_QUEUE_CREATE_INFO,
			queueFamilyIndex = queue_family,
			queueCount = 1,
			pQueuePriorities = &queue_priority,
		}
	}

	device_features: vk.PhysicalDeviceFeatures

	create_info := vk.DeviceCreateInfo {
		sType = .DEVICE_CREATE_INFO, 
		queueCreateInfoCount = u32(len(queue_create_infos)),
		pQueueCreateInfos = raw_data(queue_create_infos),
		pEnabledFeatures = &device_features,
		enabledExtensionCount = u32(len(device_extensions)),
		ppEnabledExtensionNames = raw_data(device_extensions),
	}

	when ODIN_DEBUG {
		create_info.enabledLayerCount = u32(len(validation_layers))
		create_info.ppEnabledLayerNames = raw_data(validation_layers)
	} 

	vk_check_result(vk.CreateDevice(renderer.physical_device, &create_info, 
		                            nil, &renderer.device))

	vk.GetDeviceQueue(renderer.device, indices.graphics_family.?, 0, &renderer.graphics_queue)
	assert(renderer.graphics_queue != nil) 
	vk.GetDeviceQueue(renderer.device, indices.present_family.?, 0, &renderer.present_queue)
	assert(renderer.present_queue != nil) 
}

create_swap_chain :: proc() {
	swap_chain_support := query_swap_chain_support(renderer.physical_device)

	surface_format := choose_swap_surface_format(swap_chain_support.formats)
	present_mode := choose_swap_present_mode(swap_chain_support.present_modes)
	extent := choose_swap_chain_extent(swap_chain_support.capabilities)

	image_count := swap_chain_support.capabilities.minImageCount + 1
	if swap_chain_support.capabilities.maxImageCount > 0 && 
	   image_count > swap_chain_support.capabilities.maxImageCount {
		image_count = swap_chain_support.capabilities.maxImageCount	
	}

	create_info := vk.SwapchainCreateInfoKHR {
		sType = .SWAPCHAIN_CREATE_INFO_KHR,
		surface = renderer.surface,
		minImageCount = image_count,
		imageFormat = surface_format.format,
		imageExtent = extent,
		imageArrayLayers = 1,
		imageUsage = {.COLOR_ATTACHMENT},
	}

	indices := find_queue_families(renderer.physical_device)
	queue_family_indices := []u32{
		indices.graphics_family.?,
		indices.present_family.?,
	}

	if indices.graphics_family.? != indices.present_family.? {
		create_info.imageSharingMode = .CONCURRENT
		create_info.queueFamilyIndexCount = 2
		create_info.pQueueFamilyIndices = raw_data(queue_family_indices)
	} else {
		create_info.imageSharingMode = .EXCLUSIVE
		create_info.queueFamilyIndexCount = 0
		create_info.pQueueFamilyIndices = nil
	}

	create_info.preTransform = swap_chain_support.capabilities.currentTransform
	create_info.compositeAlpha = {.OPAQUE}
	create_info.presentMode = present_mode
	create_info.clipped = true

	vk_check_result(vk.CreateSwapchainKHR(renderer.device, &create_info, 
		                                  nil, &renderer.swap_chain))

	vk_check_result(vk.GetSwapchainImagesKHR(renderer.device, renderer.swap_chain, 
		                                     &image_count, nil))
	resize(&renderer.swap_chain_images, image_count)
	vk_check_result(vk.GetSwapchainImagesKHR(renderer.device, renderer.swap_chain, 
		                                     &image_count, raw_data(renderer.swap_chain_images)))

	renderer.swap_chain_image_format = surface_format.format
	renderer.swap_chain_extent = extent
}


query_swap_chain_support :: proc(physical_device: vk.PhysicalDevice) -> Swap_Chain_Support_Details {
	details: Swap_Chain_Support_Details
	vk_check_result(vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, renderer.surface, 
		                                                       &details.capabilities))
	
	format_count: u32
	vk_check_result(vk.GetPhysicalDeviceSurfaceFormatsKHR(physical_device, renderer.surface, 
		                                                  &format_count, nil))

	if format_count != 0 {
		details.formats = make([]vk.SurfaceFormatKHR, format_count)
		vk_check_result(vk.GetPhysicalDeviceSurfaceFormatsKHR(physical_device, renderer.surface, 
			                                                  &format_count, raw_data(details.formats)))
	}

	present_mode_count: u32
	vk_check_result(vk.GetPhysicalDeviceSurfacePresentModesKHR(physical_device, renderer.surface,
	                                                           &present_mode_count, nil))

	if present_mode_count != 0 {
		details.present_modes = make([]vk.PresentModeKHR, present_mode_count)
		vk_check_result(vk.GetPhysicalDeviceSurfacePresentModesKHR(physical_device, renderer.surface,
	                                                               &present_mode_count, raw_data(details.present_modes)))
	}

	return details
}

choose_swap_chain_extent :: proc(capabilities: vk.SurfaceCapabilitiesKHR) -> vk.Extent2D {
	if capabilities.currentExtent.width != U32_MAX {
		return capabilities.currentExtent
	}

	width, height := glfw.GetFramebufferSize(window_info.handle)

	actual_extent := vk.Extent2D {
		width = u32(width), height = u32(height)
	}

	actual_extent.width = clamp(actual_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width)
	actual_extent.height = clamp(actual_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)

	return actual_extent
}

choose_swap_present_mode :: proc(available_present_modes: []vk.PresentModeKHR) -> vk.PresentModeKHR {
	for available_present_mode in available_present_modes {
		if available_present_mode == .MAILBOX {
			return available_present_mode
		}
	}

	return .FIFO
}

choose_swap_surface_format :: proc(available_formats: []vk.SurfaceFormatKHR) -> vk.SurfaceFormatKHR {
	for available_format in available_formats {
		if available_format.format == .B8G8R8A8_SRGB && 
	       available_format.colorSpace == .COLORSPACE_SRGB_NONLINEAR {
			return available_format
		}
	}

	return available_formats[0]
}

create_image_views :: proc() {
	size := len(renderer.swap_chain_images)
	resize(&renderer.swap_chain_image_views, size)

	for i := 0; i < size; i += 1 {
		create_info := vk.ImageViewCreateInfo {
			sType = .IMAGE_VIEW_CREATE_INFO,	
			image = renderer.swap_chain_images[i],
			viewType = .D2,
			format = renderer.swap_chain_image_format,
			components = {
				r = .IDENTITY,
				g = .IDENTITY,
				b = .IDENTITY,
				a = .IDENTITY,
			},
			subresourceRange = {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			}
		}

		vk_check_result(vk.CreateImageView(renderer.device, &create_info, 
			                               nil, &renderer.swap_chain_image_views[i]))
	}
}

create_render_pass :: proc() {
	color_attachment := vk.AttachmentDescription {
		format = renderer.swap_chain_image_format,
		samples = {._1},
		loadOp = .CLEAR,
		storeOp = .STORE,
		stencilLoadOp = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout = .UNDEFINED,
		finalLayout = .PRESENT_SRC_KHR,
	}

	color_attachment_ref := vk.AttachmentReference {
		attachment = 0,
		layout = .COLOR_ATTACHMENT_OPTIMAL,
	}

	subpass := vk.SubpassDescription {
		pipelineBindPoint = .GRAPHICS,
		colorAttachmentCount = 1,
		pColorAttachments = &color_attachment_ref,
	}

	render_pass_info := vk.RenderPassCreateInfo {
		sType = .RENDER_PASS_CREATE_INFO,
		attachmentCount = 1,
		pAttachments = &color_attachment,
		subpassCount = 1,
		pSubpasses = &subpass,
	}

	dependecy := vk.SubpassDependency {
		srcSubpass = vk.SUBPASS_EXTERNAL,
		dstSubpass = 0,
		srcStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		srcAccessMask = {},
		dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
	}
	render_pass_info.dependencyCount = 1
	render_pass_info.pDependencies = &dependecy

	vk_check_result(vk.CreateRenderPass(renderer.device, &render_pass_info, 
		                                nil, &renderer.render_pass))
}

create_graphics_pipeline :: proc() {
    vert_shader_module := create_shader_module(vertex_shader_code)
    defer vk.DestroyShaderModule(renderer.device, vert_shader_module, nil)

    frag_shader_module := create_shader_module(fragment_shader_code)
    defer vk.DestroyShaderModule(renderer.device, frag_shader_module, nil)

	vert_shader_stage_info := vk.PipelineShaderStageCreateInfo {
		sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage = {.VERTEX},
		module = vert_shader_module,
		pName = "main",
	}

	frag_shader_stage_info := vk.PipelineShaderStageCreateInfo {
		sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage = {.FRAGMENT},
		module = frag_shader_module,
		pName = "main",
	}

	shader_stage := []vk.PipelineShaderStageCreateInfo {
		vert_shader_stage_info,
		frag_shader_stage_info,
	}

	// NOTE: The struc pVertexBindingAttributeDescriptions
	// and pVertexBindingDescriptions should be added, but
	// becuase we are passing the shaders directly, it is not needed

	dynamic_states := []vk.DynamicState {
		.VIEWPORT, .SCISSOR,
	}
	dynamic_state := vk.PipelineDynamicStateCreateInfo {
		sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = u32(len(dynamic_states)),
		pDynamicStates = raw_data(dynamic_states),
	}

	binding_descriptions := get_binding_description()
	attribute_descriptions := get_attribute_descriptions()

	vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
		sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount = 1,
		pVertexBindingDescriptions = &binding_descriptions,
		vertexAttributeDescriptionCount = u32(len(attribute_descriptions)),
		pVertexAttributeDescriptions = raw_data(attribute_descriptions[:]),
	}

	input_assembly := vk.PipelineInputAssemblyStateCreateInfo {
		sType = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology = .TRIANGLE_LIST,
		primitiveRestartEnable = false,
	}

	viewport := vk.Viewport {
		x = 0.0, y = 0.0, 
		width = f32(renderer.swap_chain_extent.width), 
		height = f32(renderer.swap_chain_extent.height),
		minDepth = 0.0, maxDepth = 1.0,
	}

	scissor := vk.Rect2D {
		offset = {0, 0},
		extent = renderer.swap_chain_extent,
	}

	viewport_state := vk.PipelineViewportStateCreateInfo {
		sType = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		pViewports = &viewport,
		scissorCount = 1,
		pScissors = &scissor,
	}

	rasterizer := vk.PipelineRasterizationStateCreateInfo {
		sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		rasterizerDiscardEnable = false,
		polygonMode = .FILL,
		lineWidth = 1.0,
		cullMode = {.BACK},
		frontFace = .CLOCKWISE,
		depthBiasEnable = false,
		depthBiasConstantFactor = 0.0,
		depthBiasClamp = 0.0,
		depthBiasSlopeFactor = 0.0,
	}

	multisampling := vk.PipelineMultisampleStateCreateInfo {
		sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		sampleShadingEnable = false,
		rasterizationSamples = {._1},
		minSampleShading = 1.0,
		pSampleMask = nil,
		alphaToOneEnable = false,
		alphaToCoverageEnable = false,
	}

	color_blend_attachment := vk.PipelineColorBlendAttachmentState {
		colorWriteMask = {.R, .G, .B, .A},
		blendEnable = false,
		srcColorBlendFactor = .ONE,
		dstColorBlendFactor = .ZERO,
		colorBlendOp = .ADD,
		srcAlphaBlendFactor = .ONE,
		dstAlphaBlendFactor = .ZERO,
		alphaBlendOp = .ADD
	}

	color_blending := vk.PipelineColorBlendStateCreateInfo {
		sType = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		logicOpEnable = false,
		logicOp = .COPY,
		attachmentCount = 1,
		pAttachments = &color_blend_attachment,
		blendConstants = {0.0, 0.0, 0.0, 0.0},
	}

	pipeline_layout_info := vk.PipelineLayoutCreateInfo {
		sType = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount = 0,
		pSetLayouts = nil,
		pushConstantRangeCount = 0,
		pPushConstantRanges = nil,
	}

	vk_check_result(vk.CreatePipelineLayout(renderer.device, &pipeline_layout_info,
	                                        nil, &renderer.pipeline_layout))

	pipeline_info := vk.GraphicsPipelineCreateInfo {
		sType = .GRAPHICS_PIPELINE_CREATE_INFO,
		stageCount = 2,
		pStages = raw_data(shader_stage),
		pVertexInputState = &vertex_input_info,
		pInputAssemblyState = &input_assembly,
		pViewportState = &viewport_state,
		pRasterizationState = &rasterizer,
		pMultisampleState = &multisampling,
		pColorBlendState = &color_blending,
		pDynamicState = &dynamic_state,
		layout = renderer.pipeline_layout,
		renderPass = renderer.render_pass,
		subpass = 0,
		basePipelineHandle = 0,
		basePipelineIndex = -1,
	}

	vk_check_result(vk.CreateGraphicsPipelines(renderer.device, 0, 1, &pipeline_info,
	                                           nil, &renderer.graphics_pipeline))
}

create_shader_module :: proc(code: []byte) -> vk.ShaderModule {
    create_info := vk.ShaderModuleCreateInfo {
        sType = .SHADER_MODULE_CREATE_INFO,
        codeSize = len(code),
        pCode = cast(^u32)raw_data(code),
    }

    shader_module: vk.ShaderModule
    vk_check_result(vk.CreateShaderModule(renderer.device, &create_info, 
		                                  nil, &shader_module))

    return shader_module
}

create_frame_buffers :: proc() {
	length := len(renderer.swap_chain_image_views)
	resize(&renderer.swap_chainbuffers, length)

	for i in 0..<length {
		attachments := []vk.ImageView {
			renderer.swap_chain_image_views[i]
		}

		frame_buffer_info := vk.FramebufferCreateInfo {
			sType = .FRAMEBUFFER_CREATE_INFO,
			renderPass = renderer.render_pass,
			attachmentCount = 1,
			pAttachments = raw_data(attachments),
			width = renderer.swap_chain_extent.width,
			height = renderer.swap_chain_extent.height,
			layers = 1,
		}

		vk_check_result(vk.CreateFramebuffer(renderer.device, &frame_buffer_info, 
			                                 nil, &renderer.swap_chainbuffers[i]))
	}
}

create_command_pool :: proc() {
	queue_family_indices := find_queue_families(renderer.physical_device)

	pool_info := vk.CommandPoolCreateInfo {
		sType = .COMMAND_POOL_CREATE_INFO,
		flags = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = queue_family_indices.graphics_family.?,
	}

	vk_check_result(vk.CreateCommandPool(renderer.device, &pool_info, 
	                                     nil, &renderer.command_pool))
}

create_vertex_buffer :: proc() {
	buffer_size := vk.DeviceSize(size_of(vertices[0]) * len(vertices))

	staging_buffer: vk.Buffer
	staging_buffer_memory: vk.DeviceMemory

	create_buffer(buffer_size, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT}, 
		          &staging_buffer, &staging_buffer_memory)

	data: rawptr
	vk_check_result(vk.MapMemory(renderer.device, staging_buffer_memory, 0, buffer_size, {}, &data))
	{
		result := mem.copy(data, raw_data(vertices), int(buffer_size))
		assert(result != nil, 
			   fmt.tprint("Failed to copy the vertex data to the mapped memory\nFile: %v - Line: %v", #file, #line))
	}
	vk.UnmapMemory(renderer.device, staging_buffer_memory)

	create_buffer(buffer_size, {.TRANSFER_DST, .VERTEX_BUFFER}, {.DEVICE_LOCAL},
		          &renderer.vertex_buffer, &renderer.vertex_buffer_memory)

	copy_buffer(staging_buffer, renderer.vertex_buffer, buffer_size)

	vk.DestroyBuffer(renderer.device, staging_buffer, nil)
	vk.FreeMemory(renderer.device, staging_buffer_memory, nil)
}

create_index_buffer :: proc() {
	buffer_size := vk.DeviceSize(size_of(indices[0]) * len(indices))

	staging_buffer: vk.Buffer
	staging_buffer_memory: vk.DeviceMemory

	create_buffer(buffer_size, {.TRANSFER_SRC}, {.HOST_VISIBLE, .HOST_COHERENT}, 
		          &staging_buffer, &staging_buffer_memory)

	data: rawptr
	vk_check_result(vk.MapMemory(renderer.device, staging_buffer_memory, 0, buffer_size, {}, &data))
	{
		result := mem.copy(data, raw_data(indices), int(buffer_size))
		assert(result != nil, 
			   fmt.tprint("Failed to copy the index data to the mapped memory\nFile: %v - Line: %v", #file, #line))
	}
	vk.UnmapMemory(renderer.device, staging_buffer_memory)

	create_buffer(buffer_size, {.TRANSFER_DST, .INDEX_BUFFER}, {.DEVICE_LOCAL},
		          &renderer.index_buffer, &renderer.index_buffer_memory)

	copy_buffer(staging_buffer, renderer.index_buffer, buffer_size)

	vk.DestroyBuffer(renderer.device, staging_buffer, nil)
	vk.FreeMemory(renderer.device, staging_buffer_memory, nil)
}

create_command_buffer :: proc() {
	resize(&renderer.command_buffers, MAX_FRAME_IN_FLIGHT)

	alloc_info := vk.CommandBufferAllocateInfo {
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool = renderer.command_pool,
		level = .PRIMARY,
		commandBufferCount = u32(len(renderer.command_buffers)),
	}

	vk_check_result(vk.AllocateCommandBuffers(renderer.device, &alloc_info, 
		                                      raw_data(renderer.command_buffers)))
}

record_command_buffer :: proc(command_buffer: vk.CommandBuffer, image_index: u32) {
	begin_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		pInheritanceInfo = nil,
	}

	vk_check_result(vk.BeginCommandBuffer(command_buffer, &begin_info))

	render_pass_info := vk.RenderPassBeginInfo {
		sType = .RENDER_PASS_BEGIN_INFO,
		renderPass = renderer.render_pass,
		framebuffer = renderer.swap_chainbuffers[image_index],
		renderArea = {
			offset = {0, 0},
			extent = renderer.swap_chain_extent,
		}
	}

	clear_color := vk.ClearValue {
		color = vk.ClearColorValue {
			float32 = {0.0, 0.0, 0.0, 1.0}
		}
	}

	render_pass_info.clearValueCount = 1
	render_pass_info.pClearValues = &clear_color

	vk.CmdBeginRenderPass(command_buffer, &render_pass_info, .INLINE)
	{
		vk.CmdBindPipeline(command_buffer, .GRAPHICS, renderer.graphics_pipeline)

		viewport := vk.Viewport {
			x = 0.0, y = 0.0,
			width = f32(renderer.swap_chain_extent.width),
			height = f32(renderer.swap_chain_extent.height),
			minDepth = 0.0,
			maxDepth = 1.0,
		}
		vk.CmdSetViewport(command_buffer, 0, 1, &viewport)

		scissor := vk.Rect2D {
			offset = {0, 0},
			extent = renderer.swap_chain_extent,
		}
		vk.CmdSetScissor(command_buffer, 0, 1, &scissor)

		vertex_buffers := []vk.Buffer{renderer.vertex_buffer}
		offsets := []vk.DeviceSize{0}
		vk.CmdBindVertexBuffers(command_buffer, 0, 1, raw_data(vertex_buffers), raw_data(offsets))
		vk.CmdBindIndexBuffer(command_buffer, renderer.index_buffer, 0, .UINT16)

		vk.CmdDrawIndexed(command_buffer, u32(len(indices)), 1, 0, 0, 0)
	}
	vk.CmdEndRenderPass(command_buffer)

	vk_check_result(vk.EndCommandBuffer(command_buffer))
}

create_sync_objects :: proc() {
	resize(&renderer.image_available_semaphores, MAX_FRAME_IN_FLIGHT)
	resize(&renderer.render_finished_semaphores, MAX_FRAME_IN_FLIGHT)
	resize(&renderer.in_flight_fences, MAX_FRAME_IN_FLIGHT)

	semaphore_info := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
	}
	
	fence_info := vk.FenceCreateInfo {
		sType = .FENCE_CREATE_INFO,
		flags = {.SIGNALED},
	}

	for i in 0 ..< MAX_FRAME_IN_FLIGHT {
		vk_check_result(vk.CreateSemaphore(renderer.device, &semaphore_info, 
			nil, &renderer.image_available_semaphores[i]))
		vk_check_result(vk.CreateSemaphore(renderer.device, &semaphore_info, 
			nil, &renderer.render_finished_semaphores[i]))
		vk_check_result(vk.CreateFence(renderer.device, &fence_info,
			nil, &renderer.in_flight_fences[i]))
	}
}

create_buffer :: proc(
	size: vk.DeviceSize, 
	usage: vk.BufferUsageFlags, 
	properties: vk.MemoryPropertyFlags,
	buffer: ^vk.Buffer,
	buffer_memory: ^vk.DeviceMemory,
) {
	buffer_info := vk.BufferCreateInfo {
		sType = .BUFFER_CREATE_INFO,
		size = vk.DeviceSize(size_of(vertices[0]) * len(vertices)),
		usage = usage,
		sharingMode = .EXCLUSIVE,
	}

	vk_check_result(vk.CreateBuffer(renderer.device, &buffer_info, nil, buffer))

	mem_requirementes: vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(renderer.device, buffer^, &mem_requirementes)

	memory_type_index := find_memory_type(mem_requirementes.memoryTypeBits, {.HOST_VISIBLE, .HOST_COHERENT})
	assert(memory_type_index != U32_MAX, "Failed to find suitable memory type")
	
	alloc_info := vk.MemoryAllocateInfo {
		sType = .MEMORY_ALLOCATE_INFO,
		allocationSize = mem_requirementes.size,
		memoryTypeIndex = memory_type_index,
	}
	vk_check_result(vk.AllocateMemory(renderer.device, &alloc_info, nil, buffer_memory))

	vk_check_result(vk.BindBufferMemory(renderer.device, buffer^, buffer_memory^, 0))
}

copy_buffer :: proc(src_buffer, dst_buffer: vk.Buffer, size: vk.DeviceSize) {
	alloc_info := vk.CommandBufferAllocateInfo {
		sType = .COMMAND_BUFFER_ALLOCATE_INFO,
		level = .PRIMARY,
		commandPool = renderer.command_pool,
		commandBufferCount = 1,
	}

	command_buffer: vk.CommandBuffer
	vk_check_result(vk.AllocateCommandBuffers(renderer.device, &alloc_info, &command_buffer))

	begin_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
		flags = {.ONE_TIME_SUBMIT},
	}
	vk_check_result(vk.BeginCommandBuffer(command_buffer, &begin_info))

	copy_region := vk.BufferCopy {
		srcOffset = 0,
		dstOffset = 0,
		size = size,
	}
	vk.CmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region)

	vk_check_result(vk.EndCommandBuffer(command_buffer))

	submit_info := vk.SubmitInfo {
		sType = .SUBMIT_INFO,
		commandBufferCount = 1,
		pCommandBuffers = &command_buffer,
	}
	vk_check_result(vk.QueueSubmit(renderer.graphics_queue, 1, &submit_info, 0))
	vk_check_result(vk.QueueWaitIdle(renderer.graphics_queue))

	vk.FreeCommandBuffers(renderer.device, renderer.command_pool, 1, &command_buffer)
}

recreate_swap_chain :: proc() {
	w, h := glfw.GetFramebufferSize(window_info.handle)
	for w == 0 || h == 0 {
		w, h = glfw.GetFramebufferSize(window_info.handle)
		glfw.WaitEvents()
	}

	vk.DeviceWaitIdle(renderer.device)

	cleanup_swap_chains()

	create_swap_chain()
	create_image_views()
	create_frame_buffers()
}

cleanup_swap_chains :: proc() {
	for frame_buffer in renderer.swap_chainbuffers {
		vk.DestroyFramebuffer(renderer.device, frame_buffer, nil)
	}
	for image_view in renderer.swap_chain_image_views {
		vk.DestroyImageView(renderer.device, image_view, nil)
	}
	vk.DestroySwapchainKHR(renderer.device, renderer.swap_chain, nil)
}

get_binding_description :: proc() -> vk.VertexInputBindingDescription {
	binding_description := vk.VertexInputBindingDescription {
		binding = 0,
		stride = size_of(Vertex),
		inputRate = .VERTEX,
	}

	return binding_description
}

get_attribute_descriptions :: proc() -> [2]vk.VertexInputAttributeDescription {
	attribute_descriptions := [2]vk.VertexInputAttributeDescription {
		// Position
		{
			binding = 0,
			location = 0,
			format = .R32G32_SFLOAT,
			offset = u32(offset_of(Vertex, pos)),
		},
		// Color 
		{
			binding = 0,
			location = 1,
			format = .R32G32B32_SFLOAT,
			offset = u32(offset_of(Vertex, color)),

		}
	}

	return attribute_descriptions
}

find_memory_type :: proc(type_filter: u32, properties: vk.MemoryPropertyFlags) -> u32 {
	mem_properties: vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(renderer.physical_device, &mem_properties)

	count := mem_properties.memoryTypeCount
	for i in 0 ..< count  {
		if (type_filter & (1 << i)) != 0 && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties {
			return i
		}
	}

	return U32_MAX
}

init_window :: proc() {
	if ok := glfw.Init(); !ok {
		fmt.eprintln("Failed to initialize glfw")
	}

	glfw.WindowHint(glfw.CLIENT_API, glfw.NO_API)
	glfw.WindowHint(glfw.RESIZABLE, glfw.TRUE)

	window_info.name = "Triangle"
	window_info.width = 800
	window_info.height = 600
	window_info.handle = glfw.CreateWindow(window_info.width, window_info.height, 
	                                  window_info.name, nil, nil)
	if window_info.handle == nil do fmt.eprintln("Failed to create the window") 
	// glfw.SetWindowUserPointer(window_info.handle, &window_info.handle)
	glfw.SetFramebufferSizeCallback(window_info.handle, framebuffer_resize_callback)
}

framebuffer_resize_callback :: proc "c" (window: glfw.WindowHandle, width, height: i32) {
	context = runtime.default_context() 
	frame_buffer_resized = true
}

draw_frame :: proc() {
	vk.WaitForFences(renderer.device, 1, &renderer.in_flight_fences[current_frame], true, U64_MAX)
	vk.ResetFences(renderer.device, 1, &renderer.in_flight_fences[current_frame])

	image_index: u32
	vk_check_swap(vk.AcquireNextImageKHR(renderer.device, renderer.swap_chain, U64_MAX, 
		                                 renderer.image_available_semaphores[current_frame], 0, &image_index))

	vk.ResetCommandBuffer(renderer.command_buffers[current_frame], nil)
	record_command_buffer(renderer.command_buffers[current_frame], image_index)

	submit_info := vk.SubmitInfo {
		sType = .SUBMIT_INFO,
	}

	wait_semaphores := []vk.Semaphore {
		renderer.image_available_semaphores[current_frame],
	}
	wait_stages := []vk.PipelineStageFlags {
		{.COLOR_ATTACHMENT_OUTPUT},
	}

	submit_info.waitSemaphoreCount = 1
	submit_info.pWaitSemaphores = raw_data(wait_semaphores)
	submit_info.pWaitDstStageMask = raw_data(wait_stages)
	submit_info.commandBufferCount = 1
	submit_info.pCommandBuffers = &renderer.command_buffers[current_frame]

	signal_semaphores := []vk.Semaphore {
		renderer.render_finished_semaphores[current_frame],
	}

	submit_info.signalSemaphoreCount = 1
	submit_info.pSignalSemaphores = raw_data(signal_semaphores)

	vk_check_result(vk.QueueSubmit(renderer.graphics_queue, 1, &submit_info, renderer.in_flight_fences[current_frame]))

	present_info := vk.PresentInfoKHR {
		sType = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores = raw_data(signal_semaphores)
	}

	swap_chain := []vk.SwapchainKHR {
		renderer.swap_chain,	
	}
	present_info.swapchainCount = 1
	present_info.pSwapchains = raw_data(swap_chain)
	present_info.pImageIndices = &image_index
	present_info.pResults = nil

	vk_check_queue(vk.QueuePresentKHR(renderer.present_queue, &present_info))

	// NOTE: Ensure the current_frame loops around after every MAX_FRAME_IN_FLIGHT
	// enqueued frames
	current_frame = (current_frame + 1) % MAX_FRAME_IN_FLIGHT
}

cleanup :: proc() {
	cleanup_swap_chains()
	vk.DestroyBuffer(renderer.device, renderer.index_buffer, nil)
	vk.FreeMemory(renderer.device, renderer.index_buffer_memory, nil)
	vk.DestroyBuffer(renderer.device, renderer.vertex_buffer, nil)
	vk.FreeMemory(renderer.device, renderer.vertex_buffer_memory, nil)
	for i in 0 ..< MAX_FRAME_IN_FLIGHT {
		vk.DestroySemaphore(renderer.device, renderer.image_available_semaphores[i], nil)
		vk.DestroySemaphore(renderer.device, renderer.render_finished_semaphores[i], nil)
		vk.DestroyFence(renderer.device, renderer.in_flight_fences[i], nil)
	}
	vk.DestroyCommandPool(renderer.device, renderer.command_pool, nil)
	vk.DestroyPipeline(renderer.device, renderer.graphics_pipeline, nil)
	vk.DestroyPipelineLayout(renderer.device, renderer.pipeline_layout, nil)
	vk.DestroyRenderPass(renderer.device, renderer.render_pass, nil)
	vk.DestroyDevice(renderer.device, nil)
	vk.DestroySurfaceKHR(renderer.instance, renderer.surface, nil)
	when ODIN_DEBUG {
		vk.DestroyDebugUtilsMessengerEXT(renderer.instance, renderer.debug_messenger, nil)
	}
	vk.DestroyInstance(renderer.instance, nil)
	glfw.DestroyWindow(window_info.handle)
	glfw.Terminate()
}

main :: proc() {
	init_window()
	init_vulkan: {
		create_instance()
		when ODIN_DEBUG {
			setup_debug_messenger()
		}
		create_surface()
		pick_physical_device()
		create_logical_device()
		create_swap_chain()
		create_image_views()
		create_render_pass()
		create_graphics_pipeline()
		create_frame_buffers()
		create_command_pool()
		create_vertex_buffer()
		create_index_buffer()
		create_command_buffer()
		create_sync_objects()
	}	
	defer cleanup()
	defer vk.DeviceWaitIdle(renderer.device)

	for !glfw.WindowShouldClose(window_info.handle) {
		glfw.PollEvents()
		draw_frame()
	}
}
