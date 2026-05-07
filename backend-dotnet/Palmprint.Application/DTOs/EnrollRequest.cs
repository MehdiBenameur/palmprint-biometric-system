namespace Palmprint.Application.DTOs;

public class EnrollRequest
{
    public Guid TenantId { get; set; }
    public string FullName { get; set; } = string.Empty;
    public string ExternalId { get; set; } = string.Empty;
    public byte[] ImageBytes { get; set; } = Array.Empty<byte>();
}