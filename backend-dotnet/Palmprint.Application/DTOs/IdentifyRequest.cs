namespace Palmprint.Application.DTOs;

public class IdentifyRequest
{
    public Guid TenantId { get; set; }

    public byte[] ImageBytes { get; set; } = Array.Empty<byte>();
}