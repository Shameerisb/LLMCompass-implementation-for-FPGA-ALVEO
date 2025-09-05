import json

def generate_alveo_u280_config():
    config = {
        "name": "Xilinx/AMD Alveo U280",
        "device_count": 1,
        "interconnect": {
            "link": {
                "name": "PCIe/Board Fabric",
                "bandwidth_per_direction_byte": 31.25e9,   # ~PCIe Gen4 x8 effective
                "bandwidth_both_directions_byte": 62.5e9,
                "latency_second": 1e-6,
                "flit_size_byte": 64,
                "header_size_byte": 16,
                "max_payload_size_byte": 4096
            },
            "link_count_per_device": 1,
            "topology": "PCIe Gen4 x8 / Card-level fabric"
        },
        "device": {
            "device_type": "FPGA (XCU280)",
            "technology_node": "16nm UltraScale+",
            "typical_kernel_clock_Hz": 300e6,
            "device_resources": {
                "look_up_tables": 1080000,
                "registers": 2607000,
                "dsp_slices": 9024,
                "block_ram_kb": 4608,
                "ultra_ram_kb": 30720,
                "internal_sram_kb": 41 * 1024  # 41 MB total on-card global buffer
            },
            "memory_protocol": "HBM2 + DDR4",
            "memory": {
                "HBM2": {
                    "total_capacity_GB": 8,
                    "total_bandwidth_GBps": 460,
                    "pseudo_channels": 32,
                    "physical_stacks": 2
                },
                "DDR4": {
                    "total_capacity_GB": 32,
                    "channels": 2,
                    "per_channel_type": "DDR4-2400",
                    "total_bandwidth_GBps": 38
                }
            },
            "io": {
                "pci_express": {
                    "supported": ["Gen3 x16", "Gen4 x8"],
                    "max_effective_bandwidth_GBps": 31.25
                },
                "network_ports": {
                    "qsfp28_ports": 2,
                    "max_line_rate_each_Gbps": 100
                },
                "transceivers": {
                    "gtx_gth_count": "board dependent",
                    "lane_rate_Gbps": "up to 32.75"
                }
            },
            "on_card_global_buffer_MB": 41,
            "power_watts": {
                "typical": 200,
                "maximum": 225
            },
            "notes": "HBM2 provides 32 pseudo-channels exposed to the PL via an AXI switch; two external DDR4 DIMMs provide additional host-side memory."
        }
    }
    return config


if __name__ == "__main__":
    u280_config = generate_alveo_u280_config()

    # Save JSON
    with open("alveo_u280_config.json", "w") as f:
        json.dump(u280_config, f, indent=2)

    print("Alveo U280 config saved to alveo_u280_config.json")
