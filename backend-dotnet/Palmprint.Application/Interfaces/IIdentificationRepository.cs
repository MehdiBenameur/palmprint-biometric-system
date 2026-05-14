using Palmprint.Domain.Entities;

namespace Palmprint.Application.Interfaces;

public interface IIdentificationRepository
{
    Task<List<PalmTemplate>> GetActiveTemplatesByTenantAsync(Guid tenantId);
}