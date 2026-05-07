using Microsoft.EntityFrameworkCore;
using Palmprint.Domain.Entities;

namespace Palmprint.Infrastructure.Persistence;

public class PalmprintDbContext : DbContext
{
	public PalmprintDbContext(DbContextOptions<PalmprintDbContext> options)
		: base(options)
	{
	}

	public DbSet<Tenant> Tenants => Set<Tenant>();
	public DbSet<User> Users => Set<User>();
	public DbSet<PalmTemplate> PalmTemplates => Set<PalmTemplate>();
	public DbSet<BiometricOperationLog> BiometricOperationLogs => Set<BiometricOperationLog>();

	protected override void OnModelCreating(ModelBuilder modelBuilder)
	{
		base.OnModelCreating(modelBuilder);

		modelBuilder.Entity<Tenant>(entity =>
		{
			entity.HasKey(x => x.Id);

			entity.Property(x => x.Name)
				.IsRequired()
				.HasMaxLength(150);

			entity.Property(x => x.ApiKeyHash)
				.IsRequired();

			entity.HasMany(x => x.Users)
				.WithOne(x => x.Tenant)
				.HasForeignKey(x => x.TenantId)
				.OnDelete(DeleteBehavior.Restrict);
		});

		modelBuilder.Entity<User>(entity =>
		{
			entity.HasKey(x => x.Id);

			entity.Property(x => x.FullName)
				.IsRequired()
				.HasMaxLength(150);

			entity.Property(x => x.ExternalId)
				.IsRequired()
				.HasMaxLength(100);

			entity.HasIndex(x => new { x.TenantId, x.ExternalId })
				.IsUnique();

			entity.HasMany(x => x.PalmTemplates)
				.WithOne(x => x.User)
				.HasForeignKey(x => x.UserId)
				.OnDelete(DeleteBehavior.Restrict);
		});

		modelBuilder.Entity<PalmTemplate>(entity =>
		{
			entity.HasKey(x => x.Id);

			entity.Property(x => x.EncryptedEmbedding)
				.IsRequired();

			entity.Property(x => x.EmbeddingHash)
				.IsRequired();

			entity.Property(x => x.ModelVersion)
				.IsRequired()
				.HasMaxLength(50);

			entity.Property(x => x.TemplateVersion)
				.IsRequired()
				.HasMaxLength(50);

			entity.HasIndex(x => x.TenantId);
			entity.HasIndex(x => x.UserId);
		});

		modelBuilder.Entity<BiometricOperationLog>(entity =>
		{
			entity.HasKey(x => x.Id);

			entity.Property(x => x.OperationType)
				.IsRequired()
				.HasMaxLength(50);

			entity.Property(x => x.Message)
				.HasMaxLength(500);

			entity.Property(x => x.IpAddress)
				.HasMaxLength(100);

			entity.HasIndex(x => x.TenantId);
			entity.HasIndex(x => x.UserId);
		});
	}
}