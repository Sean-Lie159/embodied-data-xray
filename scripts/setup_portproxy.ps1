# setup_portproxy.ps1 —— 部署机打通"同事→本机 8787 代理"的端口转发（不改代理）
#
# 用途：让局域网同事能访问你机器 8787 端口的 hy3 代理，而代理本身仍只听
#       127.0.0.1（零改动）。用 Windows 自带 portproxy 把外部进来的 8787
#       转发到本机回环，并放行防火墙。
#
# 用法（部署机，Windows）：
#   右键"以管理员身份运行 PowerShell"，执行：
#       powershell -ExecutionPolicy Bypass -File scripts\setup_portproxy.ps1
#   或带端口参数：
#       powershell -ExecutionPolicy Bypass -File scripts\setup_portproxy.ps1 -Port 8787
#
# 该脚本幂等（可重复跑）；代理进程无需重启。撤销见文档第 6.4 节
# （docs/便携分发包设计.md）。

param([int]$Port = 8787)

# 非管理员时自动提权（弹 UAC），提权后以原路径+参数重跑。
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "需要管理员权限，正在请求提权..."
    Start-Process powershell -Verb RunAs -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy Bypass",
        "-File", "`"$PSCommandPath`"",
        "-Port", "$Port"
    )
    exit
}

Write-Host "== 建立端口转发：0.0.0.0:$Port -> 127.0.0.1:$Port =="
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$Port connectaddress=127.0.0.1 connectport=$Port

Write-Host "== 放行防火墙入站 TCP $Port =="
netsh advfirewall firewall add rule name="hy3-proxy-$Port" dir=in action=allow protocol=TCP localport=$Port

Write-Host ""
Write-Host "完成。当前 v4tov4 转发规则如下（应含 listenport=$Port）："
netsh interface portproxy show v4tov4
Write-Host ""
Write-Host "下一步：用另一台机器浏览器开 http://<你的内网IP>:$Port/v1/models"
Write-Host "  返回 401/403（未授权但有内容）＝端口已通；无法访问＝未通（见文档 6.3）。"
