# ============================================================
# fpga_mix12 虚拟引脚约束
# ------------------------------------------------------------
# 说明：
# 1. 这份 XDC 仅用于“虚拟查看/占位”，不是面向真实开发板的正式管脚分配。
# 2. 所有封装脚都从器件 xc7vx485tffg1157-1 的可用 package pin 中顺序挑选。
# 3. 这里只做 PACKAGE_PIN 绑定和一个基础时钟约束，不强加 IOSTANDARD，
#    避免在未知板卡电压域条件下引入额外不必要的 DRC。
# ============================================================

set fpga_mix12_pins {
    U17 U16 AF25 AN33 AN34 AK34 AL34 AP32 AP33 AK32 AK33 AM32 AN32 AL33 AM33 AP30 AP31 AJ30 AK31 AM30 AN30 AJ29 AK29 AL31 AM31 AL29 AL30 AK28 AL28 AK26 AK27 AJ26 AJ27 AG25 AH25 AH24 AJ25 AL25 AL26 AM26 AM27 AN29 AP29 AM25 AN25 AM28 AN28 AP25 AP26 AN27 AP27 AF24 AC27 AF33 AF34 AD34 AE34 AG32 AH32 AH34 AJ34 AJ31 AJ32 AG33 AH33 AE32 AE33 AC30 AD30 AG30 AH30 AD31 AD32 AF30 AG31 AE31 AF31 AD29 AE29 AE28 AF29 AC28 AC29 AE27 AF28 AG28 AH29 AH27 AH28 AD26 AD27 AG26 AG27 AE26 AF26 AE23
}

set fpga_mix12_ports {
    clk
    rst_n
    start
    load_en
    {load_addr[0]} {load_addr[1]} {load_addr[2]} {load_addr[3]} {load_addr[4]} {load_addr[5]} {load_addr[6]} {load_addr[7]} {load_addr[8]} {load_addr[9]} {load_addr[10]} {load_addr[11]} {load_addr[12]} {load_addr[13]} {load_addr[14]} {load_addr[15]} {load_addr[16]} {load_addr[17]}
    {load_data[0]} {load_data[1]} {load_data[2]} {load_data[3]} {load_data[4]} {load_data[5]} {load_data[6]} {load_data[7]} {load_data[8]} {load_data[9]} {load_data[10]} {load_data[11]}
    busy
    done
    out_valid
    {out_class[0]} {out_class[1]} {out_class[2]} {out_class[3]} {out_class[4]} {out_class[5]} {out_class[6]} {out_class[7]}
    {out_score[0]} {out_score[1]} {out_score[2]} {out_score[3]} {out_score[4]} {out_score[5]} {out_score[6]} {out_score[7]}
    {out_score[8]} {out_score[9]} {out_score[10]} {out_score[11]} {out_score[12]} {out_score[13]} {out_score[14]} {out_score[15]}
    {out_score[16]} {out_score[17]} {out_score[18]} {out_score[19]} {out_score[20]} {out_score[21]} {out_score[22]} {out_score[23]}
    {out_score[24]} {out_score[25]} {out_score[26]} {out_score[27]} {out_score[28]} {out_score[29]} {out_score[30]} {out_score[31]}
    {out_score[32]} {out_score[33]} {out_score[34]} {out_score[35]} {out_score[36]} {out_score[37]} {out_score[38]} {out_score[39]}
    {out_score[40]} {out_score[41]} {out_score[42]} {out_score[43]} {out_score[44]} {out_score[45]} {out_score[46]} {out_score[47]}
}

if {[llength $fpga_mix12_pins] != [llength $fpga_mix12_ports]} {
    puts "ERROR: pin/port count mismatch in fpga_mix12_virtual_pins.xdc"
}

for {set i 0} {$i < [llength $fpga_mix12_ports]} {incr i} {
    set_property PACKAGE_PIN [lindex $fpga_mix12_pins $i] [get_ports [lindex $fpga_mix12_ports $i]]
}

create_clock -period 10.000 -name sys_clk [get_ports clk]
