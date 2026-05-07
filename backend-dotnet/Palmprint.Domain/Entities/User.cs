namespace Palmprint.Domain.Entities;

public class User : BaseEntity
{
    public Guid TenantId { get; set; }
    public Tenant? Tenant { get; set; }

    public string FullName { get; set; } = string.Empty;
    public string ExternalId { get; set; } = string.Empty;

    public ICollection<PalmTemplate> PalmTemplates { get; set; } = new List<PalmTemplate>();
}