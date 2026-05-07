namespace Palmprint.Domain.Entities;

public class PalmTemplate : BaseEntity
{
    public Guid UserId { get; set; }
    public User? User { get; set; }

    public Guid TenantId { get; set; }

    public byte[] EncryptedEmbedding { get; set; } = Array.Empty<byte>();
    public string EmbeddingHash { get; set; } = string.Empty;

    public string ModelVersion { get; set; } = string.Empty;
    public double QualityScore { get; set; }

    public bool IsActive { get; set; } = true;
}