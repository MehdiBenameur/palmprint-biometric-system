namespace Palmprint.Domain.Entities;

public class BiometricOperationLog : BaseEntity
{
    public Guid TenantId { get; set; }
    public Guid? UserId { get; set; }

    public string OperationType { get; set; } = string.Empty;
    public bool Success { get; set; }

    public double? SimilarityScore { get; set; }
    public string? Message { get; set; }

    public string? IpAddress { get; set; }
}