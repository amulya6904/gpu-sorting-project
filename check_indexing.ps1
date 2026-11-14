$ErrorActionPreference = 'Stop'
# CPU simulation of launch indexing only; this does not execute CUDA.
foreach ($n in @(1000, 257)) {
    $threadsPerBlock = 256
    $blocks = [int][Math]::Ceiling($n / [double]$threadsPerBlock)
    $visits = New-Object int[] $n
    $result = New-Object int[] $n
    $skipped = 0
    for ($block = 0; $block -lt $blocks; $block++) {
        for ($thread = 0; $thread -lt $threadsPerBlock; $thread++) {
            $i = $block * $threadsPerBlock + $thread
            if ($i -lt $n) {
                $result[$i] = $i + 2 * $i
                $visits[$i]++
            } else { $skipped++ }
        }
    }
    for ($i = 0; $i -lt $n; $i++) {
        if ($visits[$i] -ne 1 -or $result[$i] -ne 3 * $i) {
            throw "FAIL: CPU indexing check n=$n index=$i"
        }
    }
    Write-Output "PASS: CPU indexing check n=$n; blocks=$blocks; threads/block=$threadsPerBlock; valid=$n; guarded extra threads=$skipped. GPU execution NOT tested."
}
